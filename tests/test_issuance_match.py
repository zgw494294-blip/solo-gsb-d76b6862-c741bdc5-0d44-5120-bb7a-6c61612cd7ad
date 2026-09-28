"""End-to-end tests for the issuance-match add-on of POST /v1/issuance/check-csr."""

from __future__ import annotations

import datetime as dt
import ipaddress

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, rsa
from cryptography.x509.oid import NameOID
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

UTC = dt.timezone.utc
NOW = dt.datetime(2026, 9, 20, tzinfo=UTC)
CHECK_AT = "2026-09-28T00:00:00Z"

CHECK_ORDER = ["public_key", "san", "issuer", "signature", "validity"]


def _name(cn):
    return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)])


def make_csr(names, *, key=None, cn="www.example.com"):
    key = key or ec.generate_private_key(ec.SECP256R1())
    general = [x509.DNSName(n) if isinstance(n, str) else n for n in names]
    csr = (
        x509.CertificateSigningRequestBuilder()
        .subject_name(_name(cn))
        .add_extension(x509.SubjectAlternativeName(general), critical=False)
        .sign(key, hashes.SHA256())
    )
    return key, csr


def make_issuer(cn="Test Issuer CA", *, key=None):
    key = key or rsa.generate_private_key(65537, 2048)
    cert = (
        x509.CertificateBuilder()
        .subject_name(_name(cn))
        .issuer_name(_name(cn))
        .public_key(key.public_key())
        .serial_number(0x1111)
        .not_valid_before(NOW - dt.timedelta(days=10))
        .not_valid_after(NOW + dt.timedelta(days=3650))
        .sign(key, hashes.SHA256())
    )
    return key, cert


def make_leaf(
    csr_key,
    issuer_key,
    issuer_name,
    names,
    *,
    not_before=NOW - dt.timedelta(days=1),
    not_after=NOW + dt.timedelta(days=90),
    serial=0x2222,
    subject="www.example.com",
    signing_key=None,
    with_san=True,
):
    builder = (
        x509.CertificateBuilder()
        .subject_name(_name(subject))
        .issuer_name(issuer_name)
        .public_key(csr_key.public_key())
        .serial_number(serial)
        .not_valid_before(not_before)
        .not_valid_after(not_after)
    )
    if with_san:
        general = [
            x509.DNSName(n) if isinstance(n, str) else n for n in names
        ]
        builder = builder.add_extension(
            x509.SubjectAlternativeName(general), critical=False
        )
    return builder.sign(signing_key or issuer_key, hashes.SHA256())


def pem(obj):
    return obj.public_bytes(serialization.Encoding.PEM).decode()


def body(csr, leaf, issuer, at=CHECK_AT, **extra):
    out = {
        "csr": pem(csr),
        "ca": "letsencrypt.org",
        "records": [
            {
                "type": "CAA",
                "owner": "example.com",
                "flag": 0,
                "tag": "issue",
                "value": "letsencrypt.org",
            }
        ],
        "leaf_certificate": pem(leaf),
        "issuer_certificate": pem(issuer),
        "verification_time": at,
    }
    out.update(extra)
    return out


def check(payload):
    return client.post("/v1/issuance/check-csr", json=payload)


def checks_map(data):
    return {c["check"]: c for c in data["issuance_match"]["checks"]}


# ---------- happy paths ----------

def test_full_match_rsa_issuer_ec_leaf():
    csr_key, csr = make_csr(["WWW.Example.COM.", "api.example.com"])
    ca_key, issuer = make_issuer()
    leaf = make_leaf(
        csr_key,
        ca_key,
        issuer.subject,
        ["api.example.com", "www.example.com"],  # order differs, sets equal
    )
    r = check(body(csr, leaf, issuer))
    assert r.status_code == 200
    data = r.json()
    assert data["authorized"] is True
    im = data["issuance_match"]
    assert im["matches"] is True
    assert [c["check"] for c in im["checks"]] == CHECK_ORDER
    assert all(c["matches"] for c in im["checks"])


def test_full_match_ec_issuer_rsa_leaf():
    csr_key, csr = make_csr(["www.example.com"], key=rsa.generate_private_key(65537, 2048))
    ca_key, issuer = make_issuer(key=ec.generate_private_key(ec.SECP256R1()))
    leaf = make_leaf(csr_key, ca_key, issuer.subject, ["www.example.com"])
    r = check(body(csr, leaf, issuer))
    assert r.status_code == 200
    assert r.json()["issuance_match"]["matches"] is True


def test_wildcard_san_normalization_matches():
    csr_key, csr = make_csr(["*.example.com"])
    ca_key, issuer = make_issuer()
    leaf = make_leaf(csr_key, ca_key, issuer.subject, ["*.EXAMPLE.com."])
    r = check(body(csr, leaf, issuer))
    assert r.status_code == 200
    assert r.json()["issuance_match"]["matches"] is True


def test_original_response_shape_has_no_issuance_match():
    _, csr = make_csr(["www.example.com"])
    r = check(
        {
            "csr": pem(csr),
            "ca": "letsencrypt.org",
            "records": [
                {
                    "type": "CAA",
                    "owner": "example.com",
                    "flag": 0,
                    "tag": "issue",
                    "value": "letsencrypt.org",
                }
            ],
        }
    )
    assert r.status_code == 200
    assert "issuance_match" not in r.json()


def test_issuance_match_independent_of_caa_authorization():
    # CAA denies the CA, but the issued certificate still conforms to the CSR.
    csr_key, csr = make_csr(["www.example.com"])
    ca_key, issuer = make_issuer()
    leaf = make_leaf(csr_key, ca_key, issuer.subject, ["www.example.com"])
    r = check(body(csr, leaf, issuer, ca="pki.goog"))
    assert r.status_code == 200
    data = r.json()
    assert data["authorized"] is False
    assert data["issuance_match"]["matches"] is True


# ---------- all-or-nothing fields (422) ----------

@pytest.mark.parametrize(
    "drop",
    ["leaf_certificate", "issuer_certificate", "verification_time"],
)
def test_partial_issuance_fields_are_422(drop):
    _, csr = make_csr(["www.example.com"])
    ca_key, issuer = make_issuer()
    payload = body(csr, issuer, issuer)  # leaf value irrelevant, validation 422s first
    del payload[drop]
    r = check(payload)
    assert r.status_code == 422
    assert "all be provided together" in str(r.json()["detail"])


def test_naive_verification_time_is_422():
    csr_key, csr = make_csr(["www.example.com"])
    ca_key, issuer = make_issuer()
    leaf = make_leaf(csr_key, ca_key, issuer.subject, ["www.example.com"])
    r = check(body(csr, leaf, issuer, at="2026-09-28T00:00:00"))
    assert r.status_code == 422


def test_offset_timezone_accepted():
    csr_key, csr = make_csr(["www.example.com"])
    ca_key, issuer = make_issuer()
    leaf = make_leaf(csr_key, ca_key, issuer.subject, ["www.example.com"])
    r = check(body(csr, leaf, issuer, at="2026-09-28T02:00:00+02:00"))
    assert r.status_code == 200
    assert r.json()["issuance_match"]["matches"] is True


# ---------- malformed material (422) ----------

def test_malformed_leaf_pem_is_422():
    _, csr = make_csr(["www.example.com"])
    ca_key, issuer = make_issuer()
    r = check(body(csr, issuer, issuer, leaf_certificate="not a pem"))
    assert r.status_code == 422
    assert r.json()["error_type"] == "IssuanceError"
    assert "leaf certificate PEM" in r.json()["detail"]


def test_malformed_issuer_pem_is_422():
    csr_key, csr = make_csr(["www.example.com"])
    ca_key, issuer = make_issuer()
    leaf = make_leaf(csr_key, ca_key, issuer.subject, ["www.example.com"])
    r = check(body(csr, leaf, issuer, issuer_certificate="not a pem"))
    assert r.status_code == 422
    assert r.json()["error_type"] == "IssuanceError"
    assert "issuer certificate PEM" in r.json()["detail"]


def test_oversized_certificate_pem_is_422():
    _, csr = make_csr(["www.example.com"])
    ca_key, issuer = make_issuer()
    r = check(body(csr, issuer, issuer, leaf_certificate="A" * (64 * 1024 + 1)))
    assert r.status_code == 422
    assert r.json()["error_type"] == "IssuanceError"
    assert "64 KiB" in r.json()["detail"]


def test_unsupported_ed25519_leaf_signature_is_422():
    csr_key, csr = make_csr(["www.example.com"])
    ed_key = ed25519.Ed25519PrivateKey.generate()
    issuer = (
        x509.CertificateBuilder()
        .subject_name(_name("Ed CA"))
        .issuer_name(_name("Ed CA"))
        .public_key(ed_key.public_key())
        .serial_number(1)
        .not_valid_before(NOW - dt.timedelta(days=10))
        .not_valid_after(NOW + dt.timedelta(days=3650))
        .sign(ed_key, None)
    )
    leaf = (
        x509.CertificateBuilder()
        .subject_name(_name("www.example.com"))
        .issuer_name(_name("Ed CA"))
        .public_key(csr_key.public_key())
        .serial_number(2)
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName("www.example.com")]),
            critical=False,
        )
        .not_valid_before(NOW - dt.timedelta(days=1))
        .not_valid_after(NOW + dt.timedelta(days=90))
        .sign(ed_key, None)
    )
    r = check(body(csr, leaf, issuer))
    assert r.status_code == 422
    assert r.json()["error_type"] == "IssuanceError"
    assert "unsupported leaf signature algorithm OID 1.3.101.112" in r.json()["detail"]


def test_bad_csr_still_rejected_before_issuance_checks():
    _, csr = make_csr(["www.example.com"])
    ca_key, issuer = make_issuer()
    payload = body(csr, issuer, issuer)
    payload["csr"] = "not a pem"
    r = check(payload)
    assert r.status_code == 422
    assert r.json()["error_type"] == "CSRError"


# ---------- public key mismatch ----------

def test_different_public_key_fails_public_key_only():
    _, csr = make_csr(["www.example.com"])
    other_key, _ = make_csr(["www.example.com"])  # a different EC key
    ca_key, issuer = make_issuer()
    leaf = make_leaf(other_key, ca_key, issuer.subject, ["www.example.com"])
    r = check(body(csr, leaf, issuer))
    assert r.status_code == 200
    data = r.json()
    assert data["issuance_match"]["matches"] is False
    checks = checks_map(data)
    assert checks["public_key"]["matches"] is False
    assert "SubjectPublicKeyInfo" in checks["public_key"]["reason"]
    assert all(checks[c]["matches"] for c in CHECK_ORDER if c != "public_key")
    assert "public_key" in data["issuance_match"]["reason"]


# ---------- SAN conformance ----------

def test_cert_without_san_fails_even_with_cn():
    csr_key, csr = make_csr(["www.example.com"])
    ca_key, issuer = make_issuer()
    leaf = make_leaf(
        csr_key, ca_key, issuer.subject, [], with_san=False, subject="www.example.com"
    )
    r = check(body(csr, leaf, issuer))
    assert r.status_code == 200
    san = checks_map(r.json())["san"]
    assert san["matches"] is False
    assert "no subjectAltName" in san["reason"]


def test_cert_non_dns_san_fails():
    csr_key, csr = make_csr(["www.example.com"])
    ca_key, issuer = make_issuer()
    leaf = make_leaf(
        csr_key,
        ca_key,
        issuer.subject,
        [x509.IPAddress(ipaddress.ip_address("192.0.2.1"))],
    )
    r = check(body(csr, leaf, issuer))
    san = checks_map(r.json())["san"]
    assert san["matches"] is False
    assert "non-DNS" in san["reason"]


def test_extra_san_name_fails():
    csr_key, csr = make_csr(["www.example.com"])
    ca_key, issuer = make_issuer()
    leaf = make_leaf(
        csr_key,
        ca_key,
        issuer.subject,
        ["www.example.com", "extra.example.com"],
    )
    r = check(body(csr, leaf, issuer))
    san = checks_map(r.json())["san"]
    assert san["matches"] is False
    assert "unexpected extra" in san["reason"]
    assert "extra.example.com" in san["reason"]


def test_missing_san_name_fails():
    csr_key, csr = make_csr(["www.example.com", "api.example.com"])
    ca_key, issuer = make_issuer()
    leaf = make_leaf(csr_key, ca_key, issuer.subject, ["www.example.com"])
    r = check(body(csr, leaf, issuer))
    san = checks_map(r.json())["san"]
    assert san["matches"] is False
    assert "missing" in san["reason"]
    assert "api.example.com" in san["reason"]


def test_replaced_san_name_fails_both_ways():
    csr_key, csr = make_csr(["www.example.com"])
    ca_key, issuer = make_issuer()
    leaf = make_leaf(csr_key, ca_key, issuer.subject, ["other.example.com"])
    r = check(body(csr, leaf, issuer))
    san = checks_map(r.json())["san"]
    assert san["matches"] is False
    assert "missing" in san["reason"] and "unexpected extra" in san["reason"]


def test_wildcard_and_base_name_do_not_match():
    csr_key, csr = make_csr(["*.example.com"])
    ca_key, issuer = make_issuer()
    leaf = make_leaf(csr_key, ca_key, issuer.subject, ["example.com"])
    r = check(body(csr, leaf, issuer))
    assert checks_map(r.json())["san"]["matches"] is False


def test_duplicate_san_names_after_normalization_fail():
    csr_key, csr = make_csr(["www.example.com"])
    ca_key, issuer = make_issuer()
    leaf = make_leaf(
        csr_key, ca_key, issuer.subject, ["Example.com", "example.com."]
    )
    r = check(body(csr, leaf, issuer))
    san = checks_map(r.json())["san"]
    assert san["matches"] is False
    assert "repeats DNS name" in san["reason"]


def test_cn_never_backfills_san():
    # CSR asks for two names; the cert SAN only carries one even though its
    # subject CN repeats a requested name.
    csr_key, csr = make_csr(["www.example.com", "api.example.com"])
    ca_key, issuer = make_issuer()
    leaf = make_leaf(
        csr_key,
        ca_key,
        issuer.subject,
        ["www.example.com"],
        subject="api.example.com",
    )
    r = check(body(csr, leaf, issuer))
    san = checks_map(r.json())["san"]
    assert san["matches"] is False
    assert "api.example.com" in san["reason"]


# ---------- issuer name / signature ----------

def test_wrong_issuer_name_fails_issuer_and_signature():
    csr_key, csr = make_csr(["www.example.com"])
    ca_key, issuer = make_issuer()
    other_key, other = make_issuer(cn="Other CA")
    leaf = make_leaf(csr_key, ca_key, issuer.subject, ["www.example.com"])
    r = check(body(csr, leaf, other))
    data = r.json()
    assert data["issuance_match"]["matches"] is False
    checks = checks_map(data)
    assert checks["issuer"]["matches"] is False
    assert "issuer name" in checks["issuer"]["reason"]
    assert checks["signature"]["matches"] is False
    assert checks["public_key"]["matches"] is True
    assert checks["san"]["matches"] is True


def test_right_issuer_name_but_wrong_signing_key_fails_signature_only():
    csr_key, csr = make_csr(["www.example.com"])
    ca_key, issuer = make_issuer()
    rogue_key = rsa.generate_private_key(65537, 2048)
    leaf = make_leaf(
        csr_key,
        ca_key,
        issuer.subject,  # same issuer Name DER...
        ["www.example.com"],
        signing_key=rogue_key,  # ...but signed by a different key
    )
    r = check(body(csr, leaf, issuer))
    data = r.json()
    assert data["issuance_match"]["matches"] is False
    checks = checks_map(data)
    assert checks["issuer"]["matches"] is True
    assert checks["signature"]["matches"] is False
    assert "signature" in checks["signature"]["reason"]


def test_rsa_signature_algorithm_requires_rsa_issuer_key():
    csr_key, csr = make_csr(["www.example.com"])
    rsa_ca_key, rsa_issuer = make_issuer()  # RSA-signed leaf
    ec_ca_key = ec.generate_private_key(ec.SECP256R1())
    leaf = make_leaf(csr_key, rsa_ca_key, rsa_issuer.subject, ["www.example.com"])
    # An EC issuer certificate with the same subject Name DER: the name
    # check passes, but an RSA signature cannot verify under an EC key.
    ec_issuer = (
        x509.CertificateBuilder()
        .subject_name(rsa_issuer.subject)
        .issuer_name(rsa_issuer.subject)
        .public_key(ec_ca_key.public_key())
        .serial_number(0x3333)
        .not_valid_before(NOW - dt.timedelta(days=10))
        .not_valid_after(NOW + dt.timedelta(days=3650))
        .sign(ec_ca_key, hashes.SHA256())
    )
    r = check(body(csr, leaf, ec_issuer))
    data = r.json()
    checks = checks_map(data)
    assert checks["issuer"]["matches"] is True
    assert checks["signature"]["matches"] is False
    assert "RSA issuer key" in checks["signature"]["reason"]


# ---------- validity window ----------

def test_expired_leaf_fails_validity():
    csr_key, csr = make_csr(["www.example.com"])
    ca_key, issuer = make_issuer()
    leaf = make_leaf(
        csr_key,
        ca_key,
        issuer.subject,
        ["www.example.com"],
        not_before=NOW - dt.timedelta(days=200),
        not_after=NOW - dt.timedelta(days=100),
    )
    r = check(body(csr, leaf, issuer))
    item = checks_map(r.json())["validity"]
    assert item["matches"] is False
    assert "expired" in item["reason"]


def test_not_yet_valid_leaf_fails_validity():
    csr_key, csr = make_csr(["www.example.com"])
    ca_key, issuer = make_issuer()
    leaf = make_leaf(
        csr_key,
        ca_key,
        issuer.subject,
        ["www.example.com"],
        not_before=NOW + dt.timedelta(days=100),
        not_after=NOW + dt.timedelta(days=200),
    )
    r = check(body(csr, leaf, issuer))
    item = checks_map(r.json())["validity"]
    assert item["matches"] is False
    assert "not yet valid" in item["reason"]


@pytest.mark.parametrize(
    "at",
    ["2026-09-19T00:00:00Z", "2026-12-19T00:00:00Z"],  # == notBefore / notAfter
)
def test_validity_boundary_second_is_valid(at):
    csr_key, csr = make_csr(["www.example.com"])
    ca_key, issuer = make_issuer()
    leaf = make_leaf(
        csr_key,
        ca_key,
        issuer.subject,
        ["www.example.com"],
        not_before=dt.datetime(2026, 9, 19, tzinfo=UTC),
        not_after=dt.datetime(2026, 12, 19, tzinfo=UTC),
    )
    r = check(body(csr, leaf, issuer, at=at))
    assert r.status_code == 200
    assert checks_map(r.json())["validity"]["matches"] is True


# ---------- aggregate reason ----------

def test_overall_reason_lists_every_failed_check():
    # Different key + missing SAN name + expired leaf: three failures.
    _, csr = make_csr(["www.example.com"])
    other_key, _ = make_csr(["www.example.com"])
    ca_key, issuer = make_issuer()
    leaf = make_leaf(
        other_key,
        ca_key,
        issuer.subject,
        ["other.example.com"],
        not_before=NOW - dt.timedelta(days=200),
        not_after=NOW - dt.timedelta(days=100),
    )
    r = check(body(csr, leaf, issuer))
    im = r.json()["issuance_match"]
    assert im["matches"] is False
    assert "fails 3 check(s)" in im["reason"]
    for name in ("public_key", "san", "validity"):
        assert name in im["reason"]
    assert len(im["checks"]) == 5
