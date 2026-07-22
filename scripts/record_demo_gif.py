import asyncio
import os
from playwright.async_api import async_playwright

async def record_demo():
    """
    Records a 45-second demo of the SparkleForge self-healing flywheel.
    Phases:
    0-10s: Issue creation
    10-25s: OpenCode agent fix
    25-35s: CI test suite
    35-45s: PR merge & dashboard update
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(viewport={"width": 1280, "height": 720})
        page = await context.new_page()
        
        # Start recording
        await page.goto("http://localhost:8501") # Assuming Streamlit dashboard
        
        # Implementation of the recording logic would go here
        # Using page.video.path() or similar
        
        await asyncio.sleep(45)
        await browser.close()

def optimize_assets():
    """
    Uses ffmpeg and gifski to optimize the recorded video into GIF/WebP.
    """
    os.makedirs("docs", exist_ok=True)
    # Example conversion commands
    # os.system("ffmpeg -i demo.mp4 -vf 'fps=15,scale=640:-1:flags=lanczos' -c:v gif docs/demo_scenario.gif")
    # os.system("ffmpeg -i demo.mp4 -vcodec libwebp -filter:v fps=fps=20 -lossless 0 -compression_level 6 -q:v 70 -loop 0 docs/demo_scenario.webp")
    pass

if __name__ == "__main__":
    asyncio.run(record_demo())
    optimize_assets()

diff --git a/.github/ISSUE_TEMPLATE/bug_report.md b/.github/ISSUE_TEMPLATE/bug_report.md
