#!/bin/bash
set -e

echo "=== 1. Re-generating Xcode Project ==="
/opt/homebrew/bin/xcodegen

echo "=== 2. Cleaning and Building Release Target ==="
rm -rf build/
xcodebuild -scheme Aura -configuration Release -derivedDataPath ./build/DerivedData clean build

echo "=== 3. Setting Up Staging Area ==="
STAGING_DIR="build/dmg_staging"
mkdir -p "$STAGING_DIR"

# Copy built application
BUILT_APP="build/DerivedData/Build/Products/Release/Aura.app"
if [ ! -d "$BUILT_APP" ]; then
    echo "Error: Built app not found at $BUILT_APP"
    exit 1
fi
cp -R "$BUILT_APP" "$STAGING_DIR/"

echo "=== 4. Creating Fancy DMG Package ==="
# Remove old DMG if it exists
rm -f Aura.dmg

/opt/homebrew/bin/create-dmg \
  --volname "Aura" \
  --window-size 600 360 \
  --icon-size 100 \
  --icon "Aura.app" 175 140 \
  --hide-extension "Aura.app" \
  --app-drop-link 425 140 \
  "Aura.dmg" \
  "$STAGING_DIR/"

echo "=== 5. Cleaning Up Temp Files ==="
rm -rf build/

echo "=== DMG Release 0.8.0 Created Successfully: Aura.dmg ==="
