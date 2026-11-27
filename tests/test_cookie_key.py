from opentrain.server.app import Auth


def test_signin_survives_a_restart(tmp_path, monkeypatch):
    """The service password is new on every start; a browser's cookie must stay valid anyway."""
    monkeypatch.setenv("OPENTRAIN_COOKIE_KEY_FILE", str(tmp_path / "cookie.key"))
    token = Auth("password-of-the-first-run").token()
    assert Auth("password-after-a-restart").token_ok(token)
    assert (tmp_path / "cookie.key").stat().st_mode & 0o777 == 0o600


def test_a_forged_cookie_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENTRAIN_COOKIE_KEY_FILE", str(tmp_path / "cookie.key"))
    exp, _ = Auth("p").token().split(".", 1)
    assert not Auth("p").token_ok(f"{exp}.{'0' * 64}")
