"""End-to-end tests for the optional issuance-match extension of
POST /v1/issuance/check-csr.

When a check-csr request additionally carries the issued leaf
certificate, the issuer certificate and a UTC verification instant, the
CSR signature and CAA evaluation run unchanged and the response gains
``issuance_match`` / ``issuance_reason`` / ``issuance_checks``. These
tests cover the conformance checks (SPKI, SAN set, issuer name, leaf
signature, validity), the all-or-none validation and the 422 vs.
``issuance_match=false`` boundary.
"""

from __future__ import annotations

import base64
import datetime as dt
import ipaddress

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import dsa, ec, rsa
from cryptography.x509.oid import NameOID
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

UTC = dt.timezone.utc
NOT_BEFORE = dt.datetime(2026, 1, 1, tzinfo=UTC)
NOT_AFTER = dt.datetime(2027, 1, 1, tzinfo=UTC)
CHECK_AT = "2026-09-28T00:00:00Z"

CHECKS_IN_ORDER = ["public_key", "san", "issuer_name", "signature", "validity"]


def _name(cn):
    return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)])


# ---------- builders ----------

def make_ca(cn="Test CA", *, key=None, key_type="ec"):
    if key is None:
        key = (
            ec.generate_private_key(ec.SECP256R1())
            if key_type == "ec"
            else rsa.generate_private_key(65537, 2048)
        )
    cert = (
        x509.CertificateBuilder()
        .subject_name(_name(cn))
        .issuer_name(_name(cn))
        .public_key(key.public_key())
        .serial_number(1)
        .not_valid_before(NOT_BEFORE - dt.timedelta(days=30))
        .not_valid_after(NOT_AFTER + dt.timedelta(days=3650))
        .sign(key, hashes.SHA256())
    )
    return key, cert


def make_csr(names, *, key=None, cn="leaf.test", san=True):
    key = key or ec.generate_private_key(ec.SECP256R1())
    builder = x509.CertificateSigningRequestBuilder().subject_name(_name(cn))
    if san:
        general = [x509.DNSName(n) if isinstance(n, str) else n for n in names]
        builder = builder.add_extension(
            x509.SubjectAlternativeName(general), critical=False
        )
    return key, builder.sign(key, hashes.SHA256())


def make_leaf(
    issuer_key,
    issuer_name,
    leaf_key,
    *,
    san_names=("www.example.com",),
    subject="leaf.test",
    serial=2,
    not_before=NOT_BEFORE,
    not_after=NOT_AFTER,
    extra_extensions=(),
    sign_key=None,
    sign_hash=hashes.SHA256(),
):
    builder = (
        x509.CertificateBuilder()
        .subject_name(_name(subject))
        .issuer_name(issuer_name)
        .public_key(leaf_key.public_key())
        .serial_number(serial)
        .not_valid_before(not_before)
        .not_valid_after(not_after)
    )
    if san_names is not None:
        builder = builder.add_extension(
            x509.SubjectAlternativeName([x509.DNSName(n) for n in san_names]),
            critical=False,
        )
    for ext in extra_extensions:
        builder = builder.add_extension(ext, critical=False)
    return builder.sign(sign_key or issuer_key, sign_hash)


def pem(obj):
    return obj.public_bytes(serialization.Encoding.PEM).decode()


def request_body(csr, leaf, ca_cert, *, check_at=CHECK_AT, ca="letsencrypt.org",
                 records=None, drop=None):
    body = {
        "csr": pem(csr),
        "ca": ca,
        "records": records or [],
        "leaf_certificate": pem(leaf),
        "issuer_certificate": pem(ca_cert),
        "verification_time": check_at,
    }
    for field in drop or ():
        del body[field]
    return body


def check(body):
    return client.post("/v1/issuance/check-csr", json=body)


@pytest.fixture
def issued():
    ca_key, ca_cert = make_ca()
    leaf_key, csr = make_csr(["www.example.com", "api.example.com"])
    leaf = make_leaf(
        ca_key,
        ca_cert.subject,
        leaf_key,
        san_names=("api.example.com", "WWW.EXAMPLE.COM."),
    )
    return SimpleMaterials(ca_key, ca_cert, leaf_key, csr, leaf)


class SimpleMaterials:
    def __init__(self, ca_key, ca_cert, leaf_key, csr, leaf):
        self.ca_key = ca_key
        self.ca_cert = ca_cert
        self.leaf_key = leaf_key
        self.csr = csr
        self.leaf = leaf


def checks_map(data):
    return {c["check"]: c for c in data["issuance_checks"]}


# ---------- happy paths ----------

def test_matching_issuance(issued):
    r = check(request_body(issued.csr, issued.leaf, issued.ca_cert))
    assert r.status_code == 200
    data = r.json()
    assert data["issuance_match"] is True
    assert [c["check"] for c in data["issuance_checks"]] == CHECKS_IN_ORDER
    assert all(c["match"] for c in data["issuance_checks"])
    # The original CAA verdict is still present and independent.
    assert data["authorized"] is True


def test_rsa_issuer_and_rsa_leaf_key():
    ca_key, ca_cert = make_ca("RSA CA", key_type="rsa")
    leaf_key, csr = make_csr(
        ["www.example.com"], key=rsa.generate_private_key(65537, 2048)
    )
    leaf = make_leaf(ca_key, ca_cert.subject, leaf_key)
    r = check(request_body(csr, leaf, ca_cert))
    assert r.status_code == 200
    assert r.json()["issuance_match"] is True


def test_wildcard_san_matches_case_and_dot_insensitively(issued):
    key, csr = make_csr(["*.EXAMPLE.com."])
    leaf = make_leaf(
        issued.ca_key,
        issued.ca_cert.subject,
        key,
        san_names=("*.example.com",),
    )
    r = check(request_body(csr, leaf, issued.ca_cert))
    assert r.status_code == 200
    assert checks_map(r.json())["san"]["match"] is True


def test_san_order_does_not_matter_but_sets_must_equal(issued):
    # issued already uses a permuted, case-varied SAN order.
    data = check(request_body(issued.csr, issued.leaf, issued.ca_cert)).json()
    assert checks_map(data)["san"]["match"] is True


@pytest.mark.parametrize("boundary", ["2026-01-01T00:00:00Z", "2027-01-01T00:00:00Z"])
def test_validity_boundaries_are_valid(issued, boundary):
    r = check(request_body(issued.csr, issued.leaf, issued.ca_cert, check_at=boundary))
    assert r.status_code == 200
    assert checks_map(r.json())["validity"]["match"] is True


def test_verification_time_with_explicit_offset(issued):
    r = check(
        request_body(issued.csr, issued.leaf, issued.ca_cert,
                     check_at="2026-09-28T02:00:00+02:00")
    )
    assert r.status_code == 200
    assert checks_map(r.json())["validity"]["match"] is True


def test_caa_unauthorized_still_reports_issuance_match(issued):
    caa_records = [
        {"type": "CAA", "owner": "example.com", "flag": 0,
         "tag": "issue", "value": "pki.goog"}
    ]
    body = request_body(issued.csr, issued.leaf, issued.ca_cert, records=caa_records)
    r = check(body)
    assert r.status_code == 200
    data = r.json()
    assert data["authorized"] is False
    assert data["issuance_match"] is True


# ---------- public key mismatch ----------

def test_different_public_key_mismatch(issued):
    other_key = ec.generate_private_key(ec.SECP256R1())
    leaf = make_leaf(issued.ca_key, issued.ca_cert.subject, other_key)
    r = check(request_body(issued.csr, leaf, issued.ca_cert))
    data = r.json()
    assert r.status_code == 200
    assert data["issuance_match"] is False
    item = checks_map(data)["public_key"]
    assert item["match"] is False
    assert "SPKI" in item["reason"]


def test_same_named_curve_different_key_mismatch():
    ca_key, ca_cert = make_ca()
    leaf_key_a = ec.generate_private_key(ec.SECP256R1())
    leaf_key_b = ec.generate_private_key(ec.SECP256R1())
    _, csr = make_csr(["www.example.com"], key=leaf_key_a)
    leaf = make_leaf(ca_key, ca_cert.subject, leaf_key_b)
    r = check(request_body(csr, leaf, ca_cert))
    assert checks_map(r.json())["public_key"]["match"] is False


# ---------- SAN mismatches ----------

def test_san_extra_name_mismatch(issued):
    leaf = make_leaf(
        issued.ca_key,
        issued.ca_cert.subject,
        issued.leaf_key,
        san_names=("www.example.com", "api.example.com", "extra.example.com"),
    )
    item = checks_map(check(request_body(issued.csr, leaf, issued.ca_cert)).json())["san"]
    assert item["match"] is False
    assert "extra" in item["reason"]


def test_san_missing_name_mismatch(issued):
    leaf = make_leaf(
        issued.ca_key, issued.ca_cert.subject, issued.leaf_key,
        san_names=("www.example.com",),
    )
    item = checks_map(check(request_body(issued.csr, leaf, issued.ca_cert)).json())["san"]
    assert item["match"] is False
    assert "missing" in item["reason"]


def test_san_wildcard_not_equivalent_to_base(issued):
    key, csr = make_csr(["example.com"])
    leaf = make_leaf(issued.ca_key, issued.ca_cert.subject, key,
                     san_names=("*.example.com",))
    item = checks_map(check(request_body(csr, leaf, issued.ca_cert)).json())["san"]
    assert item["match"] is False


def test_certificate_without_san_mismatch_even_with_cn(issued):
    leaf = make_leaf(
        issued.ca_key, issued.ca_cert.subject, issued.leaf_key,
        san_names=None, subject="www.example.com",
    )
    item = checks_map(check(request_body(issued.csr, leaf, issued.ca_cert)).json())["san"]
    assert item["match"] is False
    assert "subjectAltName" in item["reason"]


def test_certificate_ip_san_is_mismatch_not_422(issued):
    leaf = (
        x509.CertificateBuilder()
        .subject_name(_name("leaf.test"))
        .issuer_name(issued.ca_cert.subject)
        .public_key(issued.leaf_key.public_key())
        .serial_number(9)
        .not_valid_before(NOT_BEFORE)
        .not_valid_after(NOT_AFTER)
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.DNSName("www.example.com"),
                 x509.IPAddress(ipaddress.ip_address("192.0.2.1"))]
            ),
            critical=False,
        )
        .sign(issued.ca_key, hashes.SHA256())
    )
    r = check(request_body(issued.csr, leaf, issued.ca_cert))
    assert r.status_code == 200
    assert checks_map(r.json())["san"]["match"] is False


def test_certificate_duplicate_san_names_mismatch(issued):
    leaf = make_leaf(
        issued.ca_key, issued.ca_cert.subject, issued.leaf_key,
        san_names=("www.example.com", "WWW.Example.COM", "api.example.com"),
    )
    r = check(request_body(issued.csr, leaf, issued.ca_cert))
    assert r.status_code == 200
    assert checks_map(r.json())["san"]["match"] is False


def test_certificate_illegal_dns_name_mismatch_not_422(issued):
    leaf = make_leaf(
        issued.ca_key, issued.ca_cert.subject, issued.leaf_key,
        san_names=("bad..example.com", "api.example.com"),
    )
    r = check(request_body(issued.csr, leaf, issued.ca_cert))
    assert r.status_code == 200
    assert checks_map(r.json())["san"]["match"] is False


def test_cn_never_fills_san(issued):
    # CSR SAN is api.example.com only; certificate SAN matches but its CN
    # says www.example.com — SAN equality must pass, and CN must not be
    # added on either side.
    key, csr = make_csr(["api.example.com"], cn="www.example.com")
    leaf = make_leaf(
        issued.ca_key, issued.ca_cert.subject, key,
        san_names=("api.example.com",), subject="www.example.com",
    )
    data = check(request_body(csr, leaf, issued.ca_cert)).json()
    assert checks_map(data)["san"]["match"] is True


# ---------- issuer name / signature ----------

def test_issuer_subject_mismatch(issued):
    _, other_ca = make_ca("Other CA")
    # Leaf still signed by the real CA but names a different issuer subject.
    leaf = make_leaf(
        issued.ca_key,
        other_ca.subject,
        issued.leaf_key,
        san_names=("www.example.com", "api.example.com"),
    )
    r = check(request_body(issued.csr, leaf, issued.ca_cert))
    data = r.json()
    assert r.status_code == 200
    assert data["issuance_match"] is False
    assert checks_map(data)["issuer_name"]["match"] is False


def test_signature_by_different_key_under_same_subject(issued):
    other_key = ec.generate_private_key(ec.SECP256R1())
    # Same issuer subject, different signing key: issuer name matches,
    # signature fails against the supplied issuer certificate's key.
    leaf = make_leaf(
        other_key,
        issued.ca_cert.subject,
        issued.leaf_key,
        sign_key=other_key,
        san_names=("www.example.com", "api.example.com"),
    )
    data = check(request_body(issued.csr, leaf, issued.ca_cert)).json()
    assert checks_map(data)["issuer_name"]["match"] is True
    assert checks_map(data)["signature"]["match"] is False


def test_tampered_leaf_signature_mismatch(issued):
    der = bytearray(issued.leaf.public_bytes(serialization.Encoding.DER))
    der[-4] ^= 0x01
    tampered_pem = (
        "-----BEGIN CERTIFICATE-----\n"
        + base64.encodebytes(bytes(der)).decode()
        + "-----END CERTIFICATE-----\n"
    )
    body = request_body(issued.csr, issued.leaf, issued.ca_cert)
    body["leaf_certificate"] = tampered_pem
    r = check(body)
    assert r.status_code == 200
    assert checks_map(r.json())["signature"]["match"] is False


def test_ecdsa_signature_algorithm_requires_ec_issuer_key():
    # RSA issuer key submitted where the leaf carries an ECDSA signature.
    rsa_ca_key, rsa_ca = make_ca("RSA CA", key_type="rsa")
    ec_ca_key, ec_ca = make_ca("EC CA")
    leaf_key, csr = make_csr(["www.example.com"])
    leaf = make_leaf(ec_ca_key, ec_ca.subject, leaf_key)
    r = check(request_body(csr, leaf, rsa_ca))
    assert r.status_code == 200
    data = r.json()
    assert checks_map(data)["issuer_name"]["match"] is False
    assert checks_map(data)["signature"]["match"] is False
    assert "EC issuer key" in checks_map(data)["signature"]["reason"]


# ---------- validity ----------

def test_not_yet_valid(issued):
    leaf = make_leaf(
        issued.ca_key, issued.ca_cert.subject, issued.leaf_key,
        not_before=dt.datetime(2026, 10, 1, tzinfo=UTC),
        san_names=("www.example.com", "api.example.com"),
    )
    item = checks_map(check(request_body(issued.csr, leaf, issued.ca_cert)).json())[
        "validity"
    ]
    assert item["match"] is False
    assert "notBefore" in item["reason"]


def test_expired_certificate(issued):
    leaf = make_leaf(
        issued.ca_key, issued.ca_cert.subject, issued.leaf_key,
        not_after=dt.datetime(2026, 9, 27, tzinfo=UTC),
        san_names=("www.example.com", "api.example.com"),
    )
    item = checks_map(check(request_body(issued.csr, leaf, issued.ca_cert)).json())[
        "validity"
    ]
    assert item["match"] is False
    assert "notAfter" in item["reason"]


# ---------- aggregated verdict ----------

def test_overall_reason_lists_every_failed_check(issued):
    other_key = ec.generate_private_key(ec.SECP256R1())
    leaf = make_leaf(other_key, _name("Other CA"), other_key)
    data = check(request_body(issued.csr, leaf, issued.ca_cert)).json()
    assert data["issuance_match"] is False
    failed = [c["check"] for c in data["issuance_checks"] if not c["match"]]
    assert failed
    for name_ in failed:
        assert name_ in data["issuance_reason"]
    # Every check item is always reported, even after earlier failures.
    assert [c["check"] for c in data["issuance_checks"]] == CHECKS_IN_ORDER


# ---------- malformed input (422) ----------

@pytest.mark.parametrize(
    "drop",
    ["leaf_certificate", "issuer_certificate", "verification_time"],
)
def test_partial_triple_is_422(issued, drop):
    body = request_body(issued.csr, issued.leaf, issued.ca_cert, drop=[drop])
    r = check(body)
    assert r.status_code == 422
    assert "together" in str(r.json()["detail"])


def test_two_of_three_is_422(issued):
    body = request_body(
        issued.csr,
        issued.leaf,
        issued.ca_cert,
        drop=["issuer_certificate", "verification_time"],
    )
    assert check(body).status_code == 422


def test_empty_string_pem_values_are_422(issued):
    body = request_body(issued.csr, issued.leaf, issued.ca_cert)
    body["leaf_certificate"] = ""
    assert check(body).status_code == 422


def test_null_verification_time_with_pems_is_422(issued):
    body = request_body(issued.csr, issued.leaf, issued.ca_cert)
    body["verification_time"] = None
    assert check(body).status_code == 422


def test_naive_verification_time_is_422(issued):
    body = request_body(
        issued.csr, issued.leaf, issued.ca_cert,
        check_at="2026-09-28T00:00:00",
    )
    r = check(body)
    assert r.status_code == 422


def test_bad_leaf_pem_is_422(issued):
    body = request_body(issued.csr, issued.leaf, issued.ca_cert)
    body["leaf_certificate"] = "not a certificate"
    r = check(body)
    assert r.status_code == 422
    assert r.json()["error_type"] == "IssuanceMatchError"
    assert "leaf certificate" in r.json()["detail"]


def test_bad_issuer_pem_is_422(issued):
    body = request_body(issued.csr, issued.leaf, issued.ca_cert)
    body["issuer_certificate"] = (
        "-----BEGIN CERTIFICATE-----\nAAAA\n-----END CERTIFICATE-----\n"
    )
    r = check(body)
    assert r.status_code == 422
    assert r.json()["error_type"] == "IssuanceMatchError"
    assert "issuer certificate" in r.json()["detail"]


def test_csr_pem_in_leaf_field_is_422(issued):
    body = request_body(issued.csr, issued.leaf, issued.ca_cert)
    body["leaf_certificate"] = pem(issued.csr)
    r = check(body)
    assert r.status_code == 422
    assert r.json()["error_type"] == "IssuanceMatchError"


def test_oversized_leaf_pem_is_422(issued):
    body = request_body(issued.csr, issued.leaf, issued.ca_cert)
    body["leaf_certificate"] = "x" * (64 * 1024 + 1)
    assert check(body).status_code == 422


def test_oversized_issuer_pem_is_422(issued):
    body = request_body(issued.csr, issued.leaf, issued.ca_cert)
    body["issuer_certificate"] = "x" * (64 * 1024 + 1)
    assert check(body).status_code == 422


def test_dsa_signed_leaf_is_422():
    # A DSA issuer key / leaf signature is not supported: 422 rather than false.
    dsa_key = dsa.generate_private_key(key_size=2048)
    dsa_ca_name = _name("DSA CA")
    dsa_ca = (
        x509.CertificateBuilder()
        .subject_name(dsa_ca_name)
        .issuer_name(dsa_ca_name)
        .public_key(dsa_key.public_key())
        .serial_number(1)
        .not_valid_before(NOT_BEFORE)
        .not_valid_after(NOT_AFTER + dt.timedelta(days=3650))
        .sign(dsa_key, hashes.SHA256())
    )
    leaf_key, csr = make_csr(["www.example.com"])
    leaf = make_leaf(dsa_key, dsa_ca_name, leaf_key)
    r = check(request_body(csr, leaf, dsa_ca))
    assert r.status_code == 422
    assert r.json()["error_type"] == "IssuanceMatchError"
    assert "unsupported" in r.json()["detail"]


def test_ed25519_issuer_key_is_422():
    # Neither Ed25519 leaf signatures nor Ed25519 issuer keys are accepted.
    from cryptography.hazmat.primitives.asymmetric import ed25519

    ca_key = ed25519.Ed25519PrivateKey.generate()
    ca_name = _name("Ed CA")
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(1)
        .not_valid_before(NOT_BEFORE)
        .not_valid_after(NOT_AFTER + dt.timedelta(days=3650))
        .sign(ca_key, None)
    )
    leaf_key_ec, csr = make_csr(["www.example.com"])
    leaf = (
        x509.CertificateBuilder()
        .subject_name(_name("leaf.test"))
        .issuer_name(ca_name)
        .public_key(leaf_key_ec.public_key())
        .serial_number(7)
        .not_valid_before(NOT_BEFORE)
        .not_valid_after(NOT_AFTER)
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName("www.example.com")]),
            critical=False,
        )
        .sign(ca_key, None)
    )
    r = check(request_body(csr, leaf, ca_cert))
    assert r.status_code == 422
    assert r.json()["error_type"] == "IssuanceMatchError"
    assert "unsupported" in r.json()["detail"]


def test_invalid_csr_still_422_before_match(issued):
    # The CSR self-signature check is unchanged and runs first.
    der = bytearray(issued.csr.public_bytes(serialization.Encoding.DER))
    der[-4] ^= 0x01
    bad_csr_pem = (
        "-----BEGIN CERTIFICATE REQUEST-----\n"
        + base64.encodebytes(bytes(der)).decode()
        + "-----END CERTIFICATE REQUEST-----\n"
    )
    body = request_body(issued.csr, issued.leaf, issued.ca_cert)
    body["csr"] = bad_csr_pem
    r = check(body)
    assert r.status_code == 422
    assert r.json()["error_type"] == "CSRError"


def test_csr_without_san_still_422_with_triple(issued):
    key, csr = make_csr([], san=False)
    leaf = make_leaf(issued.ca_key, issued.ca_cert.subject, key,
                     san_names=("www.example.com",))
    assert check(request_body(csr, leaf, issued.ca_cert)).status_code == 422


# ---------- backward compatibility ----------

def test_legacy_request_response_is_unchanged(issued):
    body = {"csr": pem(issued.csr), "ca": "letsencrypt.org", "records": []}
    r = check(body)
    assert r.status_code == 200
    assert set(r.json().keys()) == {
        "csr_sha256", "authorized", "reason", "identifiers"
    }
