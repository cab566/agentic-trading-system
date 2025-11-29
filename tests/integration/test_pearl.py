
import sys
import traceback

def test_pearl_integration():
    """Test pearl integration"""
    try:
        # Basic import test
        import pearl
        print(f"✓ pearl import successful")
        return True
    except Exception as e:
        print(f"✗ pearl import failed: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_pearl_integration()
    sys.exit(0 if success else 1)
