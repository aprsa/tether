"""Unit tests for the shell primitives.

These are the places where being wrong is a quoting or escaping bug rather than
a logic one: the text looks plausible, the command runs, and the output is
quietly not what was meant. Both failures pinned here have happened.
"""

import pytest

from tether.shell import run_or_abort, is_identifier, printf, remote_path

# -- paths ----------------------------------------------------------------


def test_tilde_becomes_home_because_quoting_defeats_expansion():
    # shlex.quote('~/x') -> "'~/x'", and the quotes stop the shell expanding ~.
    assert remote_path('~/envs/phoebe') == '"$HOME/envs/phoebe"'
    assert remote_path('~') == '"$HOME"'


def test_absolute_and_relative_paths_are_quoted_normally():
    assert remote_path('/opt/venv') == '/opt/venv'
    assert remote_path('/opt/my venv') == "'/opt/my venv'"


def test_tilde_user_is_quoted_rather_than_mangled():
    """`~kelly` has no $HOME equivalent; quote it and fail loudly."""
    assert remote_path('~kelly/env') == "'~kelly/env'"


def test_double_quote_context_is_escaped():
    """Inside "$HOME/...", these four characters keep their meaning."""
    got = remote_path('~/a"b$c`d\\e')
    assert got == '"$HOME/a\\"b\\$c\\`d\\\\e"'


# -- printing -------------------------------------------------------------


def test_printf_emits_one_real_newline_escape():
    """Written by hand this is `printf '%s\\n'`, and one Python layer of
    escaping too many turns it into a literal backslash-n: the shell then
    prints `\\n` and a whole line of output silently disappears into one."""
    assert printf('"$HOME"') == 'printf \'%s\\n\' "$HOME"'
    assert '\\\\' not in printf('"$HOME"')


def test_printf_takes_several_words():
    assert printf('"$A"', '"$B"') == 'printf \'%s\\n%s\\n\' "$A" "$B"'


def test_printf_words_are_left_unquoted_for_the_shell_to_expand():
    """Most of them are expansions that exist to be evaluated."""
    assert '"${CONDA_EXE:-}"' in printf('"${CONDA_EXE:-}"')


def test_printing_nothing_is_a_mistake_not_an_empty_command():
    with pytest.raises(ValueError, match='at least one word'):
        printf()


# -- failing loudly -------------------------------------------------------


def test_guard_aborts_rather_than_continuing():
    got = run_or_abort('conda activate x', 'could not activate x')
    assert got == "conda activate x || { echo 'tether: could not activate x' >&2; exit 1; }"


def test_guard_quotes_its_own_message():
    """The message is interpolated into an `echo`, so it is a quoting site."""
    assert '$(whoami)' in run_or_abort('true', 'x$(whoami)')
    assert "'tether: x$(whoami)'" in run_or_abort('true', 'x$(whoami)')


# -- variable names -------------------------------------------------------


@pytest.mark.parametrize('name', ['PATH', '_x', 'A1', 'a_b_2'])
def test_usable_variable_names(name):
    assert is_identifier(name)


@pytest.mark.parametrize('name', ['2BAD', 'has-dash', 'has space', '', 'x;y', 'a\nb'])
def test_names_that_could_smuggle_code_into_an_export(name):
    assert not is_identifier(name)
