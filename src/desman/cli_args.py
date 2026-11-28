"""Click arguments that accept trailing prose."""

from typing import Any

import click


class FreeText(click.Argument):
    def __init__(self, param_decls: list[str], **kwargs: Any) -> None:
        super().__init__(param_decls, nargs=-1, callback=self._join, **kwargs)

    @staticmethod
    def _join(
        _ctx: click.Context, _param: click.Parameter, words: tuple[str, ...] | None
    ) -> str | None:
        return " ".join(words) if words else None


def extra_args_hint() -> str:
    from desman.i18n import t

    return t('Put text with spaces in double quotes, e.g. "like this".')
