from models import PipelineState, ValidatedOutput, Status, RequestType


class OutputValidator:
    """
    Layer 8: Output Validator
    A pure Python guardrail that ensures NO row is ever written to the CSV
    with missing columns, invalid enums, or cross-field contradictions.
    """

    def __init__(self):
        self.valid_statuses = [s.value for s in Status]
        self.valid_types = [t.value for t in RequestType]

    def validate(self, state: PipelineState) -> ValidatedOutput:
        try:
            # Extract basic info
            issue = state.raw.issue if state.raw.issue else ""
            subject = state.raw.subject if state.raw.subject else ""
            company = state.raw.company if state.raw.company else ""
            row_index = state.raw.row_index

            # Default fallback if critical pieces are missing
            if not state.composed or not state.type_decision:
                return self._fallback_row(
                    issue,
                    subject,
                    company,
                    row_index,
                    RequestType.PRODUCT_ISSUE.value,
                    "missing_composed_or_type",
                )

            resp = state.composed.response
            status = state.composed.status
            area = state.composed.product_area
            req_type = state.type_decision.request_type

            # 1. Enum Validation
            if status not in self.valid_statuses:
                return self._fallback_row(
                    issue,
                    subject,
                    company,
                    row_index,
                    req_type,
                    f"invalid_status_enum_{status}",
                )

            if req_type not in self.valid_types:
                # Type must be invalid if it's missing or wrong
                req_type = RequestType.INVALID.value

            # 2. Cross-Field Consistency Check
            # If L5 forced an escalation, but L7 somehow replied -> FAIL
            if (
                state.risk
                and state.risk.force_escalate
                and status == Status.REPLIED.value
            ):
                return self._fallback_row(
                    issue,
                    subject,
                    company,
                    row_index,
                    req_type,
                    "cross_field_violation_force_escalate_but_replied",
                )

            # If risk flagged, and no corpus support, but replied -> FAIL
            if (
                state.risk
                and state.risk.risk_flag
                and state.evidence
                and state.evidence.support_status == "unsupported"
                and status == Status.REPLIED.value
            ):
                return self._fallback_row(
                    issue,
                    subject,
                    company,
                    row_index,
                    req_type,
                    "cross_field_violation_unsupported_risk_replied",
                )

            # Construct justification
            codes = []
            if state.routing:
                codes.extend(state.routing.reason_codes)
            if state.risk:
                codes.extend(state.risk.reason_codes)
            if state.evidence:
                codes.extend(state.evidence.reason_codes)
            if state.composed:
                codes.extend(state.composed.reason_codes)

            justification = " | ".join(set(codes)) if codes else "ok"

            return ValidatedOutput(
                issue=str(issue),
                subject=str(subject),
                company=str(company),
                response=str(resp),
                product_area=str(area),
                status=str(status),
                request_type=str(req_type),
                justification=justification,
                row_index=row_index,
            )

        except Exception as e:
            return self._fallback_row(
                state.raw.issue if state.raw.issue else "",
                state.raw.subject if state.raw.subject else "",
                state.raw.company if state.raw.company else "",
                state.raw.row_index,
                RequestType.PRODUCT_ISSUE.value,
                f"unhandled_exception_{e}",
            )

    def _fallback_row(
        self,
        issue: str,
        subject: str,
        company: str,
        row_index: int,
        request_type: str,
        reason: str,
    ) -> ValidatedOutput:
        safe_type = (
            request_type
            if request_type in self.valid_types
            else RequestType.PRODUCT_ISSUE.value
        )
        return ValidatedOutput(
            issue=str(issue),
            subject=str(subject),
            company=str(company),
            response="This issue has been escalated to a human support agent.",
            status=Status.ESCALATED.value,
            product_area="",
            request_type=safe_type,
            justification=f"VALIDATION_FAILURE: {reason}",
            row_index=row_index,
            validation_passed=False,
            fallback_applied=True,
        )
