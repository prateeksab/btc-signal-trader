#!/usr/bin/env python3
"""
Collector Service

Runs trade intensity analysis every 5 seconds in the background.
This service processes raw trade data to generate intensity metrics.
"""

import sys
import os
import time
import logging

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data.processors.trade_intensity import TradeIntensityAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CollectorService:
    """Background service that runs trade intensity analysis"""

    def __init__(self, interval_seconds=60):
        self.interval = interval_seconds
        self.analyzer = TradeIntensityAnalyzer()
        self.running = False
        self.iteration_count = 0

    def run_once(self):
        """Run one iteration of intensity analysis"""
        try:
            metrics = self.analyzer.run_once()
            self.iteration_count += 1

            # Log summary every 10 iterations (every ~50 seconds)
            if self.iteration_count % 10 == 0:
                logger.info(
                    f"Intensity #{self.iteration_count}: "
                    f"TPS={float(metrics['trades_per_sec_60s']):.2f}, "
                    f"Aggression={float(metrics['aggression_score']):.2f}, "
                    f"Velocity={float(metrics['velocity_change']) if metrics['velocity_change'] else 0:.2%}"
                )

        except Exception as e:
            logger.error(f"Error in intensity analysis: {e}")

    def run(self):
        """Run continuous intensity analysis"""
        self.running = True
        logger.info(f"CollectorService started (every {self.interval}s)")
        logger.info("Press Ctrl+C to stop")

        try:
            while self.running:
                self.run_once()
                time.sleep(self.interval)

        except KeyboardInterrupt:
            logger.info("\nStopping CollectorService...")
            logger.info(f"Total iterations: {self.iteration_count}")
            self.running = False

    def stop(self):
        """Stop the service"""
        self.running = False


def main():
    """Run the collector service"""
    service = CollectorService(interval_seconds=60)
    service.run()


if __name__ == "__main__":
    main()
