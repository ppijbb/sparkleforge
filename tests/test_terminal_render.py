from src.core.roadmap.terminal_render import png_to_data_url, render_transcript_to_png


def test_render_plain_text_produces_valid_png():
    png = render_transcript_to_png("hello world\n")
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_ansi_colored_text_produces_valid_png():
    # green "OK" via SGR codes, as a real pty-captured status line would look
    png = render_transcript_to_png("\x1b[32m[OK] docker: ok\x1b[0m\n")
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_png_to_data_url_shape():
    png = render_transcript_to_png("x\n")
    url = png_to_data_url(png)
    assert url.startswith("data:image/png;base64,")
