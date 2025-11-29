
import sys
import traceback

def test_qlib_integration():
    """Test qlib integration"""
    try:
        # Basic import test
        import qlib
        print(f"✓ qlib import successful")
        return True
    except Exception as e:
        print(f"✗ qlib import failed: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_qlib_integration()
    sys.exit(0 if success else 1)
