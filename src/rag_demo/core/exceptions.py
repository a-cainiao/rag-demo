"""Domain exceptions translated into API responses."""


class RagError(Exception):
    """Base error for expected RAG application failures."""

    status_code = 500
    code = "rag_error"

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class InvalidInputError(RagError):
    status_code = 422
    code = "invalid_input"


class NotFoundError(RagError):
    status_code = 404
    code = "not_found"


class ConfigurationError(RagError):
    status_code = 503
    code = "configuration_error"


class UpstreamServiceError(RagError):
    status_code = 502
    code = "upstream_service_error"
