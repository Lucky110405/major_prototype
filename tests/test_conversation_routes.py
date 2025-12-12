from fastapi.testclient import TestClient
import pytest
from api.main import app
from run import setup_components


@pytest.fixture(scope='module', autouse=True)
def setup_app():
    setup_components()
    yield


def test_create_conversation_fallback_returns_id():
    client = TestClient(app)
    resp = client.post('/agents/conversations', json={'title': 'Test - fallback'})
    assert resp.status_code == 200
    body = resp.json()
    assert 'conversation' in body
    conv = body['conversation']
    assert isinstance(conv, dict)
    assert 'id' in conv and conv['id']


def test_generate_streamed_message_includes_conv_and_assistant_message():
    client = TestClient(app)
    # Use stream endpoint with no conversation_id => should create fallback conv and stream results
    resp = client.post('/agents/messages/generate/stream', json={'content': 'say hello'})
    assert resp.status_code == 200

    # Read the streaming content as raw text events.
    # The test client returns a generator - iterate and parse SSE chunks for the final event
    events = []
    for chunk in resp.iter_lines():
        if chunk.strip():
            # chunk may look like 'data: {"type":"partial"...}'
            if chunk.startswith('data: '):
                import json
                try:
                    # chunk is bytes or str; ensure str
                    c = chunk.decode('utf-8') if isinstance(chunk, (bytes, bytearray)) else chunk
                    data = json.loads(c[len('data: '):])
                except Exception:
                    continue
                events.append(data)
    # Find final event
    final_event = None
    for evt in events:
        if evt.get('type') == 'final':
            final_event = evt
            break
    assert final_event is not None
    assert final_event.get('conversation_id')
    assert final_event.get('assistant_message') is not None
