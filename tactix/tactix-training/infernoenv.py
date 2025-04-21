import asyncio
from pyppeteer import launch

async def main():
    # Launch a headless browser
    browser = await launch(
        headless=True,
        args=[
            '--no-sandbox',
            '--disable-setuid-sandbox',
            '--disable-dev-shm-usage',
            '--disable-gpu',
            '--disable-software-rasterizer'
        ]
    )
    page = await browser.newPage()

    # Open your React app (assuming it's running locally)
    await page.goto('http://localhost:8080')  # Adjust this if your React app is hosted elsewhere

    # Wait for the React app to fully load and WebSocket to connect (use any visible element for this)
    await page.waitForSelector('body')  # Wait until the body is loaded, or use a more appropriate selector

    print("App is running indefinitely, waiting for events...")

    # Infinite loop to keep the script running indefinitely
    while True:
        # For older Pyppeteer versions use waitFor(), for newer use page.waitForTimeout()
        try:
            await page.waitFor(5000)  # 5 seconds, adjust this as needed
        except Exception as e:
            print(f"Error: {e}")
        
        print("Still waiting...")

    # Close the browser
    await browser.close()

# Run the event loop
asyncio.get_event_loop().run_until_complete(main())