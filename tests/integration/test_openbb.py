
import sys
import traceback

def test_openbb_integration():
    """Test openbb integration"""
    try:
        # Basic import test
        import openbb
        print(f"✓ openbb import successful")
        return True
    except Exception as e:
        print(f"✗ openbb import failed: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_openbb_integration()
    sys.exit(0 if success else 1)
