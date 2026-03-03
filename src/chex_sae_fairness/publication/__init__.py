from .spec import PublicationSpec


def run_core_publication_pipeline(*args, **kwargs):
    from .core_pipeline import run_core_publication_pipeline as _impl

    return _impl(*args, **kwargs)


def run_supplement_publication_pipeline(*args, **kwargs):
    from .supplement_pipeline import run_supplement_publication_pipeline as _impl

    return _impl(*args, **kwargs)

__all__ = [
    "PublicationSpec",
    "run_core_publication_pipeline",
    "run_supplement_publication_pipeline",
]
