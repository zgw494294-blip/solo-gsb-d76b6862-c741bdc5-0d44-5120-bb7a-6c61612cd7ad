"""Issued-certificate vs. CSR conformance check for POST /v1/issuance/check-csr.

When a check-csr request optionally carries the already-issued leaf
certificate, the issuer certificate and a UTC verification instant, the
CSR self-signature and CAA evaluation run exactly as before and this
module additionally decides whether the certificate actually conforms
to the accepted CSR — and to the named issuer — without any network
access or trust anchor:

* the certificate SubjectPublicKeyInfo DER must equal the CSR SPKI DER
  byte for byte (same key, same algorithm encoding);
* the certificate subjectAltName must be present, carry only DNSName
  entries, and its DNSName set — normalized exactly like the CSR SAN:
  case-insensitive, one trailing dot stripped, leftmost ``*.`` wildcard
  marker preserved — must equal the CSR set (no missing, duplicate or
  extra names; the subject CN is never consulted);
* the leaf certificate issuer Name must equal the issuer certificate
  subject Name (DER byte comparison);
* the leaf signature must verify with the issuer certificate public key
  using RSA PKCS#1 v1.5 or ECDSA (SHA-1/SHA-2 family);
* the verification instant must lie within the certificate validity
  window (the boundary instants count as valid).

Malformed input — unparsable PEM or an unsupported leaf signature
algorithm / issuer key type — raises :class:`IssuanceMatchError`, a
422-grade :class:`~app.engine.CAAError`. Everything that parses but
does not agree (key, SAN, issuer name, signature, dates) is a
*non-conformance*: the check entry reports ``match=False`` with a
reason. The conclusion binds only the supplied issuer; no certificate
chain is built or trusted.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from cryptography import x509
from cryptography.exceptions import InvalidSignature, UnsupportedAlgorithm
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa
from cryptography.x509.oid import ExtensionOID

from .csr import normalize_san_dns_name
from .engine import CAAError, InvalidNameError

MAX_PEM_BYTES = 64 * 1024

# Leaf certificate signature algorithms accepted, keyed by their OID.
# RSA is PKCS#1 v1.5 only (RSASSA-PSS is not used by X.509 CAs here);
# DSA, Ed25519/Ed448 and everything else is rejected as unsupported.
_RSA_SIG_HASHES: dict[str, type[hashes.HashAlgorithm]] = {
    "1.2.840.113549.1.1.5": hashes.SHA1,
    "1.2.840.113549.1.1.11": hashes.SHA256,
    "1.2.840.113549.1.1.12": hashes.SHA384,
    "1.2.840.113549.1.1.13": hashes.SHA512,
    "1.2.840.113549.1.1.14": hashes.SHA224,
}
_ECDSA_SIG_HASHES: dict[str, type[hashes.HashAlgorithm]] = {
    "1.2.840.10045.4.1": hashes.SHA1,
    "1.2.840.10045.4.3.1": hashes.SHA224,
    "1.2.840.10045.4.3.2": hashes.SHA256,
    "1.2.840.10045.4.3.3": hashes.SHA384,
    "1.2.840.10045.4.3.4": hashes.SHA512,
}


class IssuanceMatchError(CAAError):
    """Malformed issuance-match material (PEM/algorithm); maps to 422."""


@dataclass(frozen=True)
class IssuanceCheck:
    """Outcome of one conformance check item."""

    check: str
    match: bool
    reason: str

    def as_dict(self) -> dict:
        return {"check": self.check, "match": self.match, "reason": self.reason}


@dataclass(frozen=True)
class IssuanceVerdict:
    """Overall certificate/CSR conformance verdict plus per-item reasons."""

    issuance_match: bool
    issuance_reason: str
    checks: tuple[IssuanceCheck, ...]


def _utc(dt: datetime) -> datetime:
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise IssuanceMatchError(
            "verification time carries no timezone (UTC offset required)"
        )
    return dt.astimezone(timezone.utc)


def _rfc3339(dt: datetime) -> str:
    text = dt.isoformat()
    if text.endswith("+00:00"):
        text = text[:-6] + "Z"
    return text


def _load_certificate(pem: str, *, label: str) -> x509.Certificate:
    if len(pem.encode("utf-8")) > MAX_PEM_BYTES:
        raise IssuanceMatchError(
            f"{label} exceeds {MAX_PEM_BYTES} bytes ({MAX_PEM_BYTES // 1024} KiB)"
        )
    try:
        return x509.load_pem_x509_certificate(pem.encode("utf-8"))
    except Exception as exc:  # ValueError, UnsupportedAlgorithm, ...
        raise IssuanceMatchError(f"invalid {label} PEM: {exc}") from exc


def _load_csr_spki_der(csr_pem: str) -> bytes:
    """SPKI DER of the (already verified) CSR, re-parsed for its public key."""
    try:
        csr = x509.load_pem_x509_csr(csr_pem.encode("utf-8"))
    except Exception as exc:
        raise IssuanceMatchError(f"invalid PEM PKCS#10 CSR: {exc}") from exc
    try:
        return csr.public_key().public_bytes(
            serialization.Encoding.DER,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    except UnsupportedAlgorithm as exc:
        raise IssuanceMatchError(f"CSR public key type is unsupported: {exc}") from exc


def _cert_dns_names(cert: x509.Certificate) -> Optional[set[str]]:
    """Normalized DNSName set of a certificate SAN, or None on any SAN defect.

    Mirrors the CSR rules: the SAN must exist once, every GeneralName
    must be a DNSName and the normalized (case-folded, single trailing
    dot stripped, ``*.`` leftmost-wildcard marker kept) names must not
    collide. Missing/duplicate SAN, a non-DNS identifier, an illegal
    DNS name or a post-normalization duplicate all make the certificate
    SAN unmatchable rather than malformed — the CSR was already fully
    validated by :func:`app.csr.inspect_csr`.
    """
    try:
        extension = cert.extensions.get_extension_for_oid(
            ExtensionOID.SUBJECT_ALTERNATIVE_NAME
        )
    except x509.ExtensionNotFound:
        return None
    except x509.DuplicateExtension:
        return None
    try:
        general_names = list(extension.value)
    except Exception:
        return None

    names: set[str] = set()
    for general in general_names:
        if not isinstance(general, x509.DNSName):
            return None
        try:
            identifier = normalize_san_dns_name(general.value)
        except (InvalidNameError, ValueError):
            return None
        if identifier.name in names:
            return None
        names.add(identifier.name)
    return names if names else None


def _check_public_key(leaf: x509.Certificate, csr_pem: str) -> IssuanceCheck:
    try:
        leaf_spki = leaf.public_key().public_bytes(
            serialization.Encoding.DER,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    except UnsupportedAlgorithm as exc:
        raise IssuanceMatchError(
            f"leaf certificate public key type is unsupported: {exc}"
        ) from exc
    csr_spki = _load_csr_spki_der(csr_pem)
    if leaf_spki == csr_spki:
        return IssuanceCheck(
            "public_key",
            True,
            "leaf certificate SPKI DER matches the CSR public key",
        )
    return IssuanceCheck(
        "public_key",
        False,
        "leaf certificate public key (SPKI DER) does not match the CSR public key",
    )


def _check_san(leaf: x509.Certificate, csr_names: set[str]) -> IssuanceCheck:
    cert_names = _cert_dns_names(leaf)
    if cert_names is None:
        return IssuanceCheck(
            "san",
            False,
            "leaf certificate subjectAltName is missing, duplicated, not "
            "decodable, carries a non-DNS identifier or illegal/duplicate "
            "DNSName; the subject CN is not used as a fallback",
        )
    if cert_names == csr_names:
        return IssuanceCheck(
            "san",
            True,
            "leaf certificate SAN DNSName set matches the CSR SAN exactly",
        )
    missing = sorted(csr_names - cert_names)
    extra = sorted(cert_names - csr_names)
    parts = []
    if missing:
        parts.append(f"missing {missing}")
    if extra:
        parts.append(f"extra {extra}")
    return IssuanceCheck(
        "san",
        False,
        "leaf certificate SAN DNSName set does not match the CSR SAN ("
        + "; ".join(parts)
        + ")",
    )


def _check_issuer_name(
    leaf: x509.Certificate, issuer: x509.Certificate
) -> IssuanceCheck:
    if leaf.issuer.public_bytes() == issuer.subject.public_bytes():
        return IssuanceCheck(
            "issuer_name",
            True,
            "leaf certificate issuer name matches the issuer certificate subject",
        )
    return IssuanceCheck(
        "issuer_name",
        False,
        "leaf certificate issuer name does not match the issuer certificate subject",
    )


def _check_signature(
    leaf: x509.Certificate, issuer: x509.Certificate
) -> IssuanceCheck:
    try:
        sig_oid = leaf.signature_algorithm_oid
    except UnsupportedAlgorithm as exc:
        raise IssuanceMatchError(
            f"unsupported leaf signature algorithm: {exc}"
        ) from exc
    issuer_pub = issuer.public_key()

    if not isinstance(issuer_pub, (rsa.RSAPublicKey, ec.EllipticCurvePublicKey)):
        raise IssuanceMatchError(
            "issuer certificate carries an unsupported public key type "
            "(only RSA and EC keys are accepted)"
        )

    if sig_oid.dotted_string in _RSA_SIG_HASHES:
        if not isinstance(issuer_pub, rsa.RSAPublicKey):
            return IssuanceCheck(
                "signature",
                False,
                f"leaf signature algorithm {sig_oid.dotted_string} requires an "
                "RSA issuer key",
            )

        def _verify() -> None:
            issuer_pub.verify(
                leaf.signature,
                leaf.tbs_certificate_bytes,
                padding.PKCS1v15(),
                _RSA_SIG_HASHES[sig_oid.dotted_string](),
            )
    elif sig_oid.dotted_string in _ECDSA_SIG_HASHES:
        if not isinstance(issuer_pub, ec.EllipticCurvePublicKey):
            return IssuanceCheck(
                "signature",
                False,
                f"leaf signature algorithm {sig_oid.dotted_string} requires an "
                "EC issuer key",
            )

        def _verify() -> None:
            issuer_pub.verify(
                leaf.signature,
                leaf.tbs_certificate_bytes,
                ec.ECDSA(_ECDSA_SIG_HASHES[sig_oid.dotted_string]()),
            )
    else:
        raise IssuanceMatchError(
            "unsupported leaf signature algorithm OID "
            f"{sig_oid.dotted_string} (only RSA PKCS#1 v1.5 and ECDSA are accepted)"
        )

    # The certificate DER parsed, so anything that goes wrong checking
    # the signature itself (wrong signer, malformed ECDSA integers, ...)
    # is a signature mismatch rather than malformed input.
    try:
        _verify()
    except (InvalidSignature, ValueError, TypeError) as exc:
        return IssuanceCheck(
            "signature",
            False,
            f"leaf certificate signature does not verify with the issuer "
            f"certificate ({exc})",
        )
    return IssuanceCheck(
        "signature",
        True,
        "leaf certificate signature verifies with the issuer certificate "
        "(RSA PKCS#1 v1.5 / ECDSA)",
    )


def _check_validity(leaf: x509.Certificate, check_at: datetime) -> IssuanceCheck:
    not_before = leaf.not_valid_before_utc
    not_after = leaf.not_valid_after_utc
    if check_at < not_before:
        return IssuanceCheck(
            "validity",
            False,
            f"verification time {_rfc3339(check_at)} is before the leaf "
            f"certificate notBefore {_rfc3339(not_before)}",
        )
    if check_at > not_after:
        return IssuanceCheck(
            "validity",
            False,
            f"verification time {_rfc3339(check_at)} is after the leaf "
            f"certificate notAfter {_rfc3339(not_after)} (certificate expired)",
        )
    return IssuanceCheck(
        "validity",
        True,
        f"verification time {_rfc3339(check_at)} is within the leaf certificate "
        f"validity ({_rfc3339(not_before)} to {_rfc3339(not_after)})",
    )


def match_issuance(
    *,
    csr_pem: str,
    csr_names: set[str],
    leaf_pem: str,
    issuer_pem: str,
    verification_time: datetime,
) -> IssuanceVerdict:
    """Compare an issued leaf certificate against the verified CSR and issuer.

    ``csr_names`` is the set of normalized SAN DNSNames already extracted
    from the CSR (``*.`` wildcard markers kept). Runs every check even
    after an earlier mismatch and returns one :class:`IssuanceCheck`
    per item. Raises :class:`IssuanceMatchError` for malformed PEM or an
    unsupported algorithm (HTTP 422); parseable but non-conforming
    material yields ``issuance_match=False``.
    """
    check_at = _utc(verification_time)
    leaf = _load_certificate(leaf_pem, label="leaf certificate")
    issuer = _load_certificate(issuer_pem, label="issuer certificate")

    checks = (
        _check_public_key(leaf, csr_pem),
        _check_san(leaf, csr_names),
        _check_issuer_name(leaf, issuer),
        _check_signature(leaf, issuer),
        _check_validity(leaf, check_at),
    )
    failures = [c.check for c in checks if not c.match]
    if not failures:
        reason = (
            "issued leaf certificate matches the CSR public key and SAN "
            "DNSName set, is signed by the supplied issuer and is within "
            "its validity period at the verification time; no chain trust "
            "is established"
        )
    else:
        reason = (
            "leaf certificate does not conform to the CSR / supplied issuer; "
            f"failed check(s): {failures}"
        )
    return IssuanceVerdict(
        issuance_match=not failures,
        issuance_reason=reason,
        checks=checks,
    )
