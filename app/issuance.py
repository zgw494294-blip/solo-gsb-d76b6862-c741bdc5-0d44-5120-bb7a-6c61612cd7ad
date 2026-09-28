"""Issued-certificate/CSR conformance check for POST /v1/issuance/check-csr.

After a CSR has passed its self-signature check and CAA evaluation, the
caller may additionally hand in the issued leaf certificate, the issuer
certificate and a UTC verification instant. The issued certificate is
then compared with the accepted CSR and its claimed issuer:

* the leaf's SubjectPublicKeyInfo DER must be byte-identical to the
  CSR's SPKI DER (the same public key was certified);
* the leaf's subjectAltName DNSName set, normalized with exactly the
  CSR's wildcard rules, must equal the CSR's set (no missing, extra,
  duplicate or non-DNS identifier; the subject CN is never consulted);
* the leaf's issuer Name (DER) must equal the issuer certificate's
  subject Name;
* the leaf signature must verify with the issuer certificate public key
  using RSA PKCS#1 v1.5 or ECDSA;
* the verification instant must lie within the leaf's validity window
  (the boundary second itself is valid).

Malformed material — a PEM that does not parse, a certificate/CSR key or
signature algorithm the library cannot handle — raises
:class:`IssuanceError`, a 422-grade :class:`~app.engine.CAAError`.
Everything else that merely does not match is a *verification failure*
(``matches=False`` with per-check reasons), never a chain conclusion:
the verdict only binds the supplied issuer certificate.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from cryptography import x509
from cryptography.exceptions import InvalidSignature, UnsupportedAlgorithm
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa
from cryptography.x509.oid import ExtensionOID

from .csr import normalize_san
from .engine import CAAError
from .ocsp import _ECDSA_SIG_HASHES, _RSA_SIG_HASHES, rfc3339

MAX_PEM_BYTES = 64 * 1024

# Fixed per-check order; every check is evaluated, nothing short-circuits.
CHECKS = ("public_key", "san", "issuer", "signature", "validity")


class IssuanceError(CAAError):
    """Malformed issued-certificate material (PEM/algorithm); maps to 422."""


def _utc(dt: datetime) -> datetime:
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise IssuanceError(
            "verification time carries no timezone (UTC offset required)"
        )
    return dt.astimezone(timezone.utc)


@dataclass(frozen=True)
class CheckItem:
    """One issuance conformance check and its outcome."""

    check: str
    matches: bool
    reason: str


@dataclass(frozen=True)
class IssuanceVerdict:
    """Overall conformance verdict plus the per-check evidence."""

    matches: bool
    reason: str
    checks: tuple[CheckItem, ...]


def _load_certificate(pem: str, *, label: str) -> x509.Certificate:
    if len(pem.encode("utf-8")) > MAX_PEM_BYTES:
        raise IssuanceError(
            f"{label} exceeds {MAX_PEM_BYTES} bytes "
            f"({MAX_PEM_BYTES // 1024} KiB)"
        )
    try:
        return x509.load_pem_x509_certificate(pem.encode("utf-8"))
    except Exception as exc:  # ValueError, UnsupportedAlgorithm, ...
        raise IssuanceError(f"invalid {label} PEM: {exc}") from exc


def _cert_public_key(cert: x509.Certificate, *, label: str):
    try:
        return cert.public_key()
    except UnsupportedAlgorithm as exc:
        raise IssuanceError(
            f"unsupported {label} public key algorithm: {exc}"
        ) from exc
    except Exception as exc:
        raise IssuanceError(f"{label} public key cannot be read: {exc}") from exc


def _spki_der(key) -> bytes:
    try:
        return key.public_bytes(
            serialization.Encoding.DER,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    except UnsupportedAlgorithm as exc:
        raise IssuanceError(f"unsupported public key algorithm: {exc}") from exc
    except Exception as exc:
        raise IssuanceError(f"public key cannot be read: {exc}") from exc


def _check_public_key(csr_spki_der: bytes, leaf: x509.Certificate) -> CheckItem:
    # The route already self-verified the CSR and hands in its SPKI DER;
    # a leaf key the library cannot expose is malformed input (422), not
    # a public-key mismatch.
    leaf_spki = _spki_der(_cert_public_key(leaf, label="leaf certificate"))
    if leaf_spki == csr_spki_der:
        return CheckItem(
            "public_key",
            True,
            "leaf certificate public key matches the CSR public key (SPKI DER)",
        )
    return CheckItem(
        "public_key",
        False,
        "leaf certificate SubjectPublicKeyInfo does not match the CSR public key",
    )


def _cert_dns_names(cert: x509.Certificate) -> tuple[set[str], str | None]:
    """Normalize the certificate SAN DNSName set.

    Returns ``(names, mismatch_reason)``: when the extension is missing,
    duplicated/undecodable, carries a non-DNS GeneralName, an illegal
    DNSName or duplicates after normalization, ``mismatch_reason``
    explains it and ``names`` is empty.
    """
    try:
        extension = cert.extensions.get_extension_for_oid(
            ExtensionOID.SUBJECT_ALTERNATIVE_NAME
        )
    except x509.ExtensionNotFound:
        return set(), (
            "leaf certificate carries no subjectAltName extension; "
            "the subject CN is not used as a fallback"
        )
    except x509.DuplicateExtension as exc:
        return set(), f"leaf certificate carries duplicate subjectAltName extensions: {exc}"
    try:
        general_names = list(extension.value)
    except Exception as exc:
        return set(), f"leaf certificate subjectAltName cannot be decoded: {exc}"

    names: set[str] = set()
    for general in general_names:
        if not isinstance(general, x509.DNSName):
            return set(), (
                "leaf certificate subjectAltName carries a non-DNS identifier "
                f"({type(general).__name__})"
            )
        try:
            raw = general.value
        except Exception as exc:
            return set(), f"leaf certificate SAN DNS name cannot be decoded: {exc}"
        try:
            name = normalize_san(raw).name
        except Exception as exc:  # IllegalNameError/CSRError on a cert name
            return set(), f"leaf certificate SAN DNS name is illegal: {exc}"
        if name in names:
            return set(), (
                f"leaf certificate subjectAltName repeats DNS name {name!r} "
                "after normalization"
            )
        names.add(name)

    if not names:
        return set(), "leaf certificate subjectAltName carries no DNSName entries"
    return names, None


def _check_san(csr_names: frozenset[str], leaf: x509.Certificate) -> CheckItem:
    leaf_names, mismatch = _cert_dns_names(leaf)
    if mismatch is not None:
        return CheckItem("san", False, mismatch)
    if leaf_names == set(csr_names):
        shown = ", ".join(sorted(leaf_names))
        return CheckItem(
            "san",
            True,
            f"normalized SAN DNSName set matches the CSR: {{{shown}}}",
        )
    missing = sorted(set(csr_names) - leaf_names)
    extra = sorted(leaf_names - set(csr_names))
    parts = []
    if missing:
        parts.append(f"missing {missing}")
    if extra:
        parts.append(f"unexpected extra {extra}")
    return CheckItem(
        "san",
        False,
        "leaf certificate SAN DNSName set does not match the CSR: "
        + "; ".join(parts),
    )


def _check_issuer(leaf: x509.Certificate, issuer: x509.Certificate) -> CheckItem:
    if leaf.issuer.public_bytes() == issuer.subject.public_bytes():
        return CheckItem(
            "issuer",
            True,
            "leaf certificate issuer name matches the issuer certificate subject",
        )
    return CheckItem(
        "issuer",
        False,
        "leaf certificate issuer name does not match the issuer certificate subject",
    )


def _check_signature(leaf: x509.Certificate, issuer: x509.Certificate) -> CheckItem:
    try:
        sig_oid = leaf.signature_algorithm_oid
    except UnsupportedAlgorithm as exc:
        raise IssuanceError(f"unsupported leaf signature algorithm: {exc}") from exc

    issuer_pub = _cert_public_key(issuer, label="issuer certificate")
    oid = sig_oid.dotted_string
    if oid in _RSA_SIG_HASHES:
        if not isinstance(issuer_pub, rsa.RSAPublicKey):
            return CheckItem(
                "signature",
                False,
                f"leaf signature algorithm {oid} requires an RSA issuer key",
            )

        def _verify() -> None:
            issuer_pub.verify(
                leaf.signature,
                leaf.tbs_certificate_bytes,
                padding.PKCS1v15(),
                _RSA_SIG_HASHES[oid](),
            )
    elif oid in _ECDSA_SIG_HASHES:
        if not isinstance(issuer_pub, ec.EllipticCurvePublicKey):
            return CheckItem(
                "signature",
                False,
                f"leaf signature algorithm {oid} requires an EC issuer key",
            )

        def _verify() -> None:
            issuer_pub.verify(
                leaf.signature,
                leaf.tbs_certificate_bytes,
                ec.ECDSA(_ECDSA_SIG_HASHES[oid]()),
            )
    else:
        raise IssuanceError(
            "unsupported leaf signature algorithm OID "
            f"{oid} (only RSA PKCS#1 v1.5 and ECDSA are accepted)"
        )

    # The certificate DER parsed, so a signature that does not verify or
    # cannot be decoded is a signature mismatch, not malformed input.
    try:
        _verify()
    except (InvalidSignature, ValueError, TypeError) as exc:
        return CheckItem(
            "signature",
            False,
            f"leaf certificate signature does not verify with the issuer "
            f"certificate ({exc})",
        )
    return CheckItem(
        "signature",
        True,
        "leaf certificate signature verifies with the issuer certificate "
        "(RSA PKCS#1 v1.5/ECDSA)",
    )


def _check_validity(leaf: x509.Certificate, check_at: datetime) -> CheckItem:
    not_before = _utc(leaf.not_valid_before_utc)
    not_after = _utc(leaf.not_valid_after_utc)
    if check_at < not_before:
        return CheckItem(
            "validity",
            False,
            f"leaf certificate is not yet valid: notBefore {rfc3339(not_before)} "
            f"is later than the verification time {rfc3339(check_at)}",
        )
    if check_at > not_after:
        return CheckItem(
            "validity",
            False,
            f"leaf certificate is expired: notAfter {rfc3339(not_after)} is "
            f"before the verification time {rfc3339(check_at)}",
        )
    return CheckItem(
        "validity",
        True,
        f"verification time {rfc3339(check_at)} lies within the leaf validity "
        f"window ({rfc3339(not_before)} .. {rfc3339(not_after)})",
    )


def match_issuance(
    csr_spki_der: bytes,
    csr_names: frozenset[str],
    leaf_pem: str,
    issuer_pem: str,
    verification_time: datetime,
) -> IssuanceVerdict:
    """Check an issued leaf certificate against its accepted CSR and issuer.

    ``csr_spki_der`` is the CSR's SubjectPublicKeyInfo DER and
    ``csr_names`` its normalized SAN DNSName set, both produced by the
    already-verified CSR inspection. Raises :class:`IssuanceError` on
    malformed PEM or unsupported algorithms (HTTP 422); every parseable
    but non-conforming input yields ``matches=False`` with one
    :class:`CheckItem` per check in :data:`CHECKS`. The conclusion only
    binds ``issuer_pem``; no chain is built or trusted.
    """
    check_at = _utc(verification_time)
    leaf = _load_certificate(leaf_pem, label="leaf certificate")
    issuer = _load_certificate(issuer_pem, label="issuer certificate")

    checks = (
        _check_public_key(csr_spki_der, leaf),
        _check_san(csr_names, leaf),
        _check_issuer(leaf, issuer),
        _check_signature(leaf, issuer),
        _check_validity(leaf, check_at),
    )
    failed = [item.check for item in checks if not item.matches]
    if not failed:
        reason = (
            "issued leaf certificate matches the CSR and is signed by and "
            "valid under the supplied issuer certificate"
        )
    else:
        reason = (
            f"issued leaf certificate fails {len(failed)} check(s): "
            + ", ".join(failed)
        )
    return IssuanceVerdict(matches=not failed, reason=reason, checks=checks)
