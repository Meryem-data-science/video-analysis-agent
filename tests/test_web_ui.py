import sys

import pytest
from fastapi.testclient import TestClient

from src.video_agent import web_api, web_cli


def _settings(tmp_path, *, with_frontend=True):
    frontend_dist = tmp_path / "frontend" / "dist"
    if with_frontend:
        (frontend_dist / "assets").mkdir(parents=True)
        (frontend_dist / "index.html").write_text(
            "<!doctype html><title>Video Agent UI</title><div id='root'></div>",
            encoding="utf-8",
        )
        (frontend_dist / "assets" / "app.js").write_text(
            "window.VIDEO_AGENT_UI=true;", encoding="utf-8"
        )
    return web_api.ApiSettings(
        project_root=tmp_path,
        runs_root=tmp_path / "outputs" / "runs",
        upload_root=tmp_path / "uploads",
        frontend_dist=frontend_dist,
        max_upload_bytes=1024,
        max_workers=1,
    )


def test_cors_allows_only_local_development_origins(tmp_path):
    client = TestClient(web_api.create_app(_settings(tmp_path, with_frontend=False)))
    allowed = client.options(
        "/api/health",
        headers={
            "Origin": "http://127.0.0.1:5173",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"
    assert allowed.headers["access-control-allow-credentials"] == "true"

    refused = client.options(
        "/api/health",
        headers={
            "Origin": "https://example.invalid",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert "access-control-allow-origin" not in refused.headers


def test_compiled_frontend_and_assets_are_served(tmp_path):
    app = web_api.create_app(_settings(tmp_path), serve_ui=True)
    client = TestClient(app)
    assert app.state.ui_available is True
    assert "Video Agent UI" in client.get("/").text
    assert client.get("/assets/app.js").text == "window.VIDEO_AGENT_UI=true;"


def test_spa_fallback_serves_index_on_direct_frontend_route(tmp_path):
    client = TestClient(web_api.create_app(_settings(tmp_path), serve_ui=True))
    response = client.get("/runs/demo/results")
    assert response.status_code == 200
    assert "Video Agent UI" in response.text


def test_api_routes_keep_priority_over_spa_fallback(tmp_path):
    client = TestClient(web_api.create_app(_settings(tmp_path), serve_ui=True))
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    unknown = client.get("/api/unknown")
    assert unknown.status_code == 404
    assert unknown.json()["error"]["code"] == "api_route_not_found"
    assert "Video Agent UI" not in unknown.text


def test_missing_compiled_frontend_does_not_break_api(tmp_path):
    app = web_api.create_app(_settings(tmp_path, with_frontend=False), serve_ui=True)
    client = TestClient(app)
    assert app.state.ui_available is False
    assert client.get("/api/health").status_code == 200
    assert client.get("/").status_code == 404


def test_web_cli_help_exposes_ui_option(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["video-agent web", "--help"])
    with pytest.raises(SystemExit) as exc:
        web_cli.main()
    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "--ui" in output
    assert "--host" in output
    assert "--port" in output


def test_web_cli_finds_frontend_build_from_current_project(monkeypatch, tmp_path):
    index = tmp_path / "frontend" / "dist" / "index.html"
    index.parent.mkdir(parents=True)
    index.write_text("<!doctype html>", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert web_cli._find_frontend_index() == index.resolve()
