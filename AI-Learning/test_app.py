
import unittest
from unittest.mock import patch, MagicMock
import chat

class TestChatInterface(unittest.TestCase):
    @patch('chat.create_session_history', return_value=[{"role": "system", "content": "Test system"}])
    @patch('chat.get_benji_response')
    def test_chat_flow(self, mock_benji_response, mock_create_history):
        # Simulate Benji response and chat history update
        mock_benji_response.return_value = ("Hello, user!", [{"role": "system", "content": "Test system"}, {"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello, user!"}])
        chat_history = mock_create_history()
        reply, updated_history = chat.get_benji_response("Hi", chat_history)
        self.assertEqual(reply, "Hello, user!")
        self.assertEqual(updated_history[-1]["role"], "assistant")

    def test_main_exit(self):
        # Patch input to simulate user typing 'exit' immediately
        with patch('builtins.input', side_effect=["exit"]), patch('builtins.print') as mock_print:
            chat.main()
            mock_print.assert_any_call("Goodbye!")

if __name__ == "__main__":
    unittest.main()
