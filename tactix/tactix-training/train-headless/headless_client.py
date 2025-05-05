#!/usr/bin/env python3
import asyncio
import logging
import os
import sys
import json
import time
from datetime import datetime
from playwright.async_api import async_playwright

# Configure logging
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("headless-client")

# Configuration from environment variables
REACT_APP_URL = os.environ.get("REACT_APP_URL", "http://react-client:8080")
SCREENSHOT_DIR = "/app/screenshots"

# Ensure screenshot directory exists
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

async def take_screenshot(page, name_prefix="screenshot"):
    """Take a screenshot for debugging purposes"""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{name_prefix}_{timestamp}.png"
    filepath = os.path.join(SCREENSHOT_DIR, filename)
    await page.screenshot(path=filepath)
    logger.info(f"Screenshot saved to {filepath}")
    return filepath

async def main():
    """Main function to run the headless browser client"""
    logger.info(f"Starting headless browser client")
    logger.info(f"Target React app: {REACT_APP_URL}/#/tactics")
    
    # Single connection attempt with proper error handling
    try:
        async with async_playwright() as p:
            # Launch browser
            browser = await p.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-setuid-sandbox", 
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                ]
            )
            
            logger.info("Browser launched successfully")
            
            # Create a new page
            page = await browser.new_page()
            
            # Set up console log forwarding
            page.on("console", lambda msg: logger.info(f"BROWSER: {msg.text}"))
            page.on("pageerror", lambda err: logger.error(f"PAGE ERROR: {err}"))
            
            # Navigate to the React app
            logger.info(f"Navigating to {REACT_APP_URL}/#/tactics")
            
            try:
                response = await page.goto(f"{REACT_APP_URL}/#/tactics", wait_until="networkidle", timeout=60000)
                
                if response and response.ok:
                    logger.info(f"Page loaded successfully: {response.status}")
                else:
                    logger.error(f"Failed to load page: {response.status if response else 'No response'}")
                    await take_screenshot(page, "load_error")
                    raise Exception(f"Failed to load page")
                    
            except Exception as e:
                logger.error(f"Navigation error: {e}")
                await take_screenshot(page, "navigation_error")
                raise
            
            # Take a screenshot to verify the page loaded
            await take_screenshot(page, "initial_load")
            
            # Execute JavaScript to check WebSocket connection status
            websocket_status = await page.evaluate("""
                () => {
                    // Log websocket status for debugging
                    if (window.simulationModel && window.simulationModel.socket) {
                        const status = window.simulationModel.socket.readyState;
                        const statusText = ['CONNECTING', 'OPEN', 'CLOSING', 'CLOSED'][status];
                        console.log('WebSocket status:', statusText);
                        return statusText;
                    } else {
                        console.log('WebSocket not found yet');
                        return 'NOT_FOUND';
                    }
                }
            """)
            logger.info(f"WebSocket status: {websocket_status}")
            
            # Keep the browser open indefinitely - maintain a single session
            logger.info("Headless browser running - maintaining session")
            
            screenshot_interval = 300  # 5 minutes
            last_screenshot_time = time.time()
            
            while True:
                try:
                    # Perform a simple check to ensure the page is still responsive
                    await page.evaluate("1 + 1")
                    
                    # Take periodic screenshots
                    current_time = time.time()
                    if current_time - last_screenshot_time >= screenshot_interval:
                        await take_screenshot(page, "status")
                        last_screenshot_time = current_time
                        
                        # Log WebSocket status periodically
                        websocket_status = await page.evaluate("""
                            () => {
                                if (window.simulationModel && window.simulationModel.socket) {
                                    const status = window.simulationModel.socket.readyState;
                                    const statusText = ['CONNECTING', 'OPEN', 'CLOSING', 'CLOSED'][status];
                                    return statusText;
                                } else {
                                    return 'NOT_FOUND';
                                }
                            }
                        """)
                        logger.info(f"Periodic check - WebSocket status: {websocket_status}")
                        
                except Exception as e:
                    logger.error(f"Page health check failed: {e}")
                    await take_screenshot(page, "error_state")
                    # Don't exit the loop, just log the error and continue
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)