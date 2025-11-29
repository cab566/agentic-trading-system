
import sys
import traceback

def test_rd_agent_integration():
    """Test rd_agent integration"""
    try:
        # Basic import test
        import rdagent
        print(f"✓ rd_agent import successful")
        return True
    except Exception as e:
        print(f"✗ rd_agent import failed: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_rd_agent_integration()
    sys.exit(0 if success else 1)
