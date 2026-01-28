#!/bin/bash

# Get parameters
PATCH_DIR=$1
BASE_DIR=$2
VERSION=$3
PWDATA_NAME=$4
PWACT_NAME=$5
CPU_ONPY=$6
# Build paths
if [ $CPU_ONPY -eq 0 ]; then
  ENV_DIR="$BASE_DIR/matpl-$VERSION"
  SITE_PACKAGES_DIR="$ENV_DIR/lib/python3.11/site-packages"
  MATPL_DIR="$BASE_DIR/MatPL-$VERSION"
else
  ENV_DIR="$BASE_DIR/matpl_cpu-$VERSION"
  SITE_PACKAGES_DIR="$ENV_DIR/lib/python3.11/site-packages"
  MATPL_DIR="$BASE_DIR/MatPL_cpu-$VERSION"
fi

# activate env

# 激活 matpl-2025.3 Python 环境
source $ENV_DIR/bin/activate

# Build full package paths
PWDATA_PACKAGE="$PWDATA_NAME"
PWACT_PACKAGE="$PWACT_NAME"

# Check if directories exist
if [ ! -d "$SITE_PACKAGES_DIR" ]; then
    echo "Error: Directory $SITE_PACKAGES_DIR does not exist"
    exit 1
fi

if [ ! -d "$PATCH_DIR" ]; then
    echo "Error: Patch directory $PATCH_DIR does not exist"
    exit 1
fi

# Install pwact package
install_pwact() {
    local OFFLINE_PACKAGE="$PWACT_PACKAGE"
    local PACKAGE="pwact"
    
    # Check if offline package exists
    if [ ! -f "$OFFLINE_PACKAGE" ]; then
        echo "Error: Offline package $OFFLINE_PACKAGE does not exist"
        return 1
    fi
    
    # Extract version from package filename
    local OFFLINE_VERSION=$(echo "$PWACT_NAME" | grep -oP '(?<=-)[0-9]+\.[0-9]+\.[0-9]+(?=\.tar\.gz)')
    
    # Check currently installed version
    local DIST_INFO_DIR="${SITE_PACKAGES_DIR}/${PACKAGE}-*.dist-info"
    if [ -d $DIST_INFO_DIR ]; then
        local INSTALLED_VERSION=$(basename $DIST_INFO_DIR | grep -oP '(?<=-)[0-9]+\.[0-9]+\.[0-9]+(?=\.dist-info)')
        
        echo "Checking package: $PACKAGE"
        echo "Installed version: $INSTALLED_VERSION"
        echo "Offline version: $OFFLINE_VERSION"
        
        # Compare versions
        if [ "$INSTALLED_VERSION" != "$OFFLINE_VERSION" ]; then
            echo "Versions differ, updating $PACKAGE..."
            pip install "$OFFLINE_PACKAGE"
            if [ $? -eq 0 ]; then
                echo "Successfully updated $PACKAGE to version $OFFLINE_VERSION"
                return 2  # Updated
            else
                echo "Failed to update $PACKAGE"
                return 1  # Failed
            fi
        else
            echo "Versions match, no update needed for $PACKAGE"
            return 0  # No update needed
        fi
    else
        echo "Package $PACKAGE not found, installing..."
        pip install "$OFFLINE_PACKAGE"
        if [ $? -eq 0 ]; then
            echo "Successfully installed $PACKAGE version $OFFLINE_VERSION"
            return 2  # Installed
        else
            echo "Failed to install $PACKAGE"
            return 1  # Failed
        fi
    fi
}

# Install pwdata package
install_pwdata() {
    local OFFLINE_PACKAGE="$PWDATA_PACKAGE"
    local PACKAGE="pwdata"
    
    # Check if offline package exists
    if [ ! -f "$OFFLINE_PACKAGE" ]; then
        echo "Error: Offline package $OFFLINE_PACKAGE does not exist"
        return 1
    fi
    
    # Extract version from package filename
    local OFFLINE_VERSION=$(echo "$PWDATA_NAME" | grep -oP '(?<=-)[0-9]+\.[0-9]+\.[0-9]+(?=\.tar\.gz)')
    
    # Check currently installed version
    local DIST_INFO_DIR="${SITE_PACKAGES_DIR}/${PACKAGE}-*.dist-info"
    if [ -d $DIST_INFO_DIR ]; then
        local INSTALLED_VERSION=$(basename $DIST_INFO_DIR | grep -oP '(?<=-)[0-9]+\.[0-9]+\.[0-9]+(?=\.dist-info)')
        
        echo "Checking package: $PACKAGE"
        echo "Installed version: $INSTALLED_VERSION"
        echo "Offline version: $OFFLINE_VERSION"
        
        # Compare versions
        if [ "$INSTALLED_VERSION" != "$OFFLINE_VERSION" ]; then
            echo "Versions differ, updating $PACKAGE..."
            pip install "$OFFLINE_PACKAGE"
            if [ $? -eq 0 ]; then
                echo "Successfully updated $PACKAGE to version $OFFLINE_VERSION"
                return 2  # Updated
            else
                echo "Failed to update $PACKAGE"
                return 1  # Failed
            fi
        else
            echo "Versions match, no update needed for $PACKAGE"
            return 0  # No update needed
        fi
    else
        echo "Package $PACKAGE not found, installing..."
        pip install "$OFFLINE_PACKAGE"
        if [ $? -eq 0 ]; then
            echo "Successfully installed $PACKAGE version $OFFLINE_VERSION"
            return 2  # Installed
        else
            echo "Failed to install $PACKAGE"
            return 1  # Failed
        fi
    fi
}

# Display configuration information
echo "=== Installation Configuration ==="
echo "Patch directory: $PATCH_DIR"
echo "Base directory: $BASE_DIR"
echo "Version: $VERSION"
echo "Environment directory: $ENV_DIR"
echo "MatPL directory: $MATPL_DIR"
echo "pwdata package: $PWDATA_PACKAGE"
echo "pwact package: $PWACT_PACKAGE"
echo "site-packages directory: $SITE_PACKAGES_DIR"
echo "========================================"

# Main execution logic
echo "Starting pwact package installation..."
echo "----------------------------------------"
install_pwact
PWACT_RESULT=$?
echo "----------------------------------------"

echo "Starting pwdata package installation..."
echo "----------------------------------------"
install_pwdata
PWDATA_RESULT=$?
echo "----------------------------------------"

# Output installation result summary
echo "Installation summary:"
if [ $PWACT_RESULT -eq 1 ]; then
    echo "✗ pwact installation failed"
elif [ $PWACT_RESULT -eq 2 ]; then
    echo "✓ pwact installation successful"
fi

if [ $PWDATA_RESULT -eq 1 ]; then
    echo "✗ pwdata installation failed"
elif [ $PWDATA_RESULT -eq 2 ]; then
    echo "✓ pwdata installation successful"
fi

if [ $PWACT_RESULT -eq 1 ] || [ $PWDATA_RESULT -eq 1 ]; then
    echo "Some packages failed to install"
    exit 1
elif [ $PWACT_RESULT -eq 0 ] && [ $PWDATA_RESULT -eq 0 ]; then
    echo "All packages are up to date, no installation needed"
    exit 0
else
    echo "Installation completed"
    exit 0
fi
