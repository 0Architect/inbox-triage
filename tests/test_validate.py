from datetime import date, timedelta

from schema import ApplicationFields, Extraction, Intent, LeadFields
from validate import validate


def _lead(**overrides) -> Extraction:
    fields = dict(
        contact_name="Jane Doe",
        contact_email="jane@example.com",
        contact_phone="(212) 555-0147",
        property_ref="123 Main St, Apt 2",
        desired_move_in=date.today() + timedelta(days=30),
    )
    fields.update(overrides)
    return Extraction(intent=Intent.new_lead, lead=LeadFields(**fields))


def _application(**overrides) -> Extraction:
    fields = dict(
        applicant_name="Jane Doe",
        desired_unit="123 Main St, Apt 2",
        monthly_income=5000.0,
        screening_consent=True,
    )
    fields.update(overrides)
    return Extraction(intent=Intent.application, application=ApplicationFields(**fields))


RAW_TEXT = (
    "Hi, my name is Jane Doe. I'm interested in 123 Main St, Apt 2. "
    "Email: jane@example.com Phone: (212) 555-0147. Move in 30 days."
)


def test_valid_lead_passes():
    result = validate(_lead(), RAW_TEXT)
    assert result.passed
    assert result.failed_checks == []


def test_invalid_email_fails():
    result = validate(_lead(contact_email="not-an-email"), RAW_TEXT)
    assert not result.passed
    assert "invalid email format" in result.failed_checks


def test_invalid_phone_fails():
    result = validate(_lead(contact_phone="123"), RAW_TEXT)
    assert not result.passed
    assert "invalid phone number" in result.failed_checks


def test_past_move_in_date_fails():
    result = validate(_lead(desired_move_in=date.today() - timedelta(days=5)), RAW_TEXT)
    assert not result.passed
    assert "move-in date invalid or in the past" in result.failed_checks


def test_hallucinated_property_ref_fails():
    result = validate(_lead(property_ref="999 Nonexistent Blvd, Nowhere, ZZ"), RAW_TEXT)
    assert not result.passed
    assert any("hallucination" in c for c in result.failed_checks)


def test_typo_tolerant_property_ref_passes():
    # Minor char-level noise (as generate_data.py injects) should still fuzzy-match.
    result = validate(_lead(property_ref="123 Man St, Apt 2"), RAW_TEXT)
    assert result.passed


def test_missing_contact_method_flagged():
    result = validate(_lead(contact_email=None, contact_phone=None), RAW_TEXT)
    assert not result.passed
    assert any("missing required fields" in c for c in result.failed_checks)


def test_valid_application_passes():
    result = validate(_application(), RAW_TEXT)
    assert result.passed
    assert result.failed_checks == []


def test_application_missing_required_fields_recorded_on_schema():
    ext = _application(monthly_income=None, screening_consent=None)
    result = validate(ext, RAW_TEXT)
    assert not result.passed
    assert "monthly_income" in ext.application.missing_required_fields
    assert "screening_consent" in ext.application.missing_required_fields


def test_no_contact_info_at_all_reports_all_required_missing():
    ext = Extraction(intent=Intent.new_lead, lead=None)
    result = validate(ext, RAW_TEXT)
    assert not result.passed
    assert any("missing required fields" in c for c in result.failed_checks)


def test_existing_tenant_has_no_field_checks():
    ext = Extraction(intent=Intent.existing_tenant, urgent_flag=True)
    result = validate(ext, "my sink is leaking, please help")
    assert result.passed
