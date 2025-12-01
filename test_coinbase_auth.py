#!/usr/bin/env python3
"""Test Coinbase API authentication and list available products"""

import os
import time
import hmac
import hashlib
import base64
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('COINBASE_API_KEY')
API_SECRET = os.getenv('COINBASE_API_SECRET')

def generate_signature(timestamp, method, path, body=''):
    """Generate HMAC SHA256 signature"""
    message = timestamp + method + path + body
    signature = hmac.new(
        base64.b64decode(API_SECRET),
        message.encode('utf-8'),
        hashlib.sha256
    ).digest()
    return base64.b64encode(signature).decode('utf-8')

def test_auth(endpoint):
    """Test authentication on a specific endpoint"""
    print(f"\nTesting endpoint: {endpoint}")

    url = f"https://api.coinbase.com{endpoint}"
    timestamp = str(int(time.time()))
    signature = generate_signature(timestamp, 'GET', endpoint)

    headers = {
        'CB-ACCESS-KEY': API_KEY,
        'CB-ACCESS-SIGN': signature,
        'CB-ACCESS-TIMESTAMP': timestamp,
        'Accept': 'application/json'
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success!")
            return data
        else:
            print(f"❌ Error: {response.text[:200]}")
            return None
    except Exception as e:
        print(f"❌ Exception: {e}")
        return None

print("="*80)
print("Coinbase API Authentication Test")
print("="*80)

# Test 1: List all products
print("\n" + "="*80)
print("TEST 1: List all available products")
print("="*80)
data = test_auth('/api/v3/brokerage/products')
if data and 'products' in data:
    products = data['products']
    print(f"\nFound {len(products)} products")

    # Look for BTC products
    btc_products = [p for p in products if 'BTC' in p.get('product_id', '')]
    print(f"\nBTC products ({len(btc_products)}):")
    for product in btc_products[:10]:
        print(f"  - {product.get('product_id')}: {product.get('product_type')}")

# Test 2: Try BTC-PERP-INTX specifically
print("\n" + "="*80)
print("TEST 2: Check BTC-PERP-INTX product")
print("="*80)
test_auth('/api/v3/brokerage/products/BTC-PERP-INTX')

# Test 3: Try BTC-PERP-INTX ticker
print("\n" + "="*80)
print("TEST 3: Check BTC-PERP-INTX ticker")
print("="*80)
test_auth('/api/v3/brokerage/products/BTC-PERP-INTX/ticker')

# Test 4: Check account info
print("\n" + "="*80)
print("TEST 4: Check account access")
print("="*80)
test_auth('/api/v3/brokerage/accounts')

print("\n" + "="*80)
print("Test Complete")
print("="*80)
