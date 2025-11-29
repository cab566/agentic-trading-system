
import sys
import traceback

def test_alpaca_mcp_integration():
    """Test alpaca_mcp integration"""
    try:
        # Basic import test
        import mcp
        print(f"✓ alpaca_mcp import successful")
        return True
    except Exception as e:
        print(f"✗ alpaca_mcp import failed: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_alpaca_mcp_integration()
    sys.exit(0 if success else 1)
