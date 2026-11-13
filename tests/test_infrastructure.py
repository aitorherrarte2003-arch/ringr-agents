"""Tests de las piezas de apoyo: configuración, auth, HTTP, idempotencia y modelos."""

from collections.abc import Callable

import pytest

from ringr_agents.auth import BearerTokenAuth
from ringr_agents.config import API_TOKEN_ENV_VAR, DEFAULT_TEST_TOKEN, Settings
from ringr_agents.conversation import Conversation, Message, Role
from ringr_agents.http_client import (
    HttpRequest,
    HttpResponse,
    HttpTransportError,
    SimulatedHttpClient,
)
from ringr_agents.idempotency import Reservation, action_scope, data_fingerprint
from ringr_agents.models import ConversationModel, ParserModel


class TestSettingsAndAuth:
    def test_token_defaults_to_test_token(self) -> None:
        assert Settings.from_env({}).api_token == DEFAULT_TEST_TOKEN

    def test_token_is_read_from_environment(self) -> None:
        assert Settings.from_env({API_TOKEN_ENV_VAR: "prod-token"}).api_token == "prod-token"

    def test_bearer_header(self) -> None:
        auth = BearerTokenAuth.from_settings(Settings(api_token="abc"))
        assert auth.auth_headers() == {"Authorization": "Bearer abc"}

    @pytest.mark.parametrize("token", ["", "   "])
    def test_empty_token_is_rejected(self, token: str) -> None:
        with pytest.raises(ValueError, match="vacío"):
            BearerTokenAuth(token)

    def test_token_never_appears_in_repr(self) -> None:
        assert "abc" not in repr(BearerTokenAuth("abc"))
        assert "abc" not in repr(Settings(api_token="abc"))


class TestHttp:
    def make_request(self) -> HttpRequest:
        return HttpRequest(
            method="POST",
            url="https://x.test",
            headers={"Authorization": "Bearer secret", "a": "1"},
            body={"a": 1},
        )

    def test_request_cannot_be_modified(self) -> None:
        request = self.make_request()
        with pytest.raises(TypeError):
            request.headers["a"] = "2"  # type: ignore[index]
        with pytest.raises(TypeError):
            request.body["a"] = 2  # type: ignore[index]

    def test_request_is_isolated_from_original_dicts(self) -> None:
        headers = {"a": "1"}
        request = HttpRequest(method="POST", url="https://x.test", headers=headers, body={})
        headers["a"] = "changed"
        assert request.headers["a"] == "1"

    def test_repr_hides_token(self) -> None:
        text = repr(self.make_request())
        assert "secret" not in text
        assert "'Authorization': '***'" in text

    def test_json_body_keeps_unicode(self) -> None:
        request = HttpRequest(method="POST", url="u", headers={}, body={"r": "información"})
        assert request.json_body() == '{"r": "información"}'

    @pytest.mark.parametrize(
        ("status", "ok"),
        [(200, True), (201, True), (299, True), (199, False), (400, False), (500, False)],
    )
    def test_response_ok(self, status: int, ok: bool) -> None:
        assert HttpResponse(status).ok is ok

    def test_simulated_client_answers_200_and_records(self) -> None:
        client = SimulatedHttpClient()
        request = self.make_request()

        response = client.send(request)

        assert response.status_code == 200
        assert client.sent_requests == (request,)

    def test_simulated_client_can_simulate_failures(self) -> None:
        def broken(request: HttpRequest) -> HttpResponse:
            raise HttpTransportError("timeout")

        client = SimulatedHttpClient(broken)
        with pytest.raises(HttpTransportError):
            client.send(self.make_request())
        assert len(client.sent_requests) == 1


class TestIdempotency:
    def test_same_data_same_fingerprint_whatever_the_key_order(self) -> None:
        assert data_fingerprint({"a": 1, "b": 2}) == data_fingerprint({"b": 2, "a": 1})

    def test_different_data_different_fingerprint(self) -> None:
        assert data_fingerprint({"a": 1}) != data_fingerprint({"a": 2})

    @pytest.mark.parametrize(("conversation", "action"), [("c2", "act"), ("c1", "other")])
    def test_each_conversation_and_action_is_its_own_scope(
        self, conversation: str, action: str
    ) -> None:
        assert action_scope("c1", "act") != action_scope(conversation, action)

    def test_the_key_depends_only_on_scope_revision_and_data(self) -> None:
        # El token (quién tiene la reserva) no cuenta: el reintento de un envío
        # lleva otro token y debe llevar la misma clave. El reintento real, a
        # través del registro, está en tests/test_ledgers.py.
        first = Reservation("scope", data_fingerprint({"a": 1}), revision=1, token="t1")
        retry = Reservation("scope", data_fingerprint({"a": 1}), revision=1, token="t2")
        assert first.idempotency_key == retry.idempotency_key

    @pytest.mark.parametrize(
        "other",
        [
            Reservation("other-scope", data_fingerprint({"a": 1}), revision=1),
            Reservation("scope", data_fingerprint({"a": 2}), revision=1),
            # Mismos datos que un envío anterior, pero es otro envío: otra clave.
            Reservation("scope", data_fingerprint({"a": 1}), revision=3),
        ],
    )
    def test_any_other_send_gets_another_key(self, other: Reservation) -> None:
        reservation = Reservation("scope", data_fingerprint({"a": 1}), revision=1)
        assert reservation.idempotency_key != other.idempotency_key


class TestModelsAndConversation:
    def test_conversation_history(self) -> None:
        conversation = Conversation("c1")
        conversation.add_user_message("hola")
        conversation.add_assistant_message("buenas")

        assert conversation.id == "c1"
        assert conversation.messages == (
            Message(Role.USER, "hola"),
            Message(Role.ASSISTANT, "buenas"),
        )

    def test_conversation_can_start_with_messages(self) -> None:
        history = [Message(Role.USER, "hola")]
        conversation = Conversation(messages=history)
        history.append(Message(Role.USER, "otro"))  # la lista original no le afecta

        assert conversation.messages == (Message(Role.USER, "hola"),)

    def test_conversation_gets_unique_id(self) -> None:
        assert Conversation().id != Conversation().id

    def test_models_follow_the_script_and_then_repeat_the_last(self) -> None:
        conversation_model = ConversationModel(["uno", "dos"])
        assert [conversation_model.answer_user() for _ in range(3)] == ["uno", "dos", "dos"]

        parser_model = ParserModel([{"a": 1}, {"a": 2}])
        assert [parser_model.parse_data() for _ in range(3)] == [{"a": 1}, {"a": 2}, {"a": 2}]

    def test_a_single_value_is_a_whole_script(self) -> None:
        # Un texto suelto es una respuesta, no una lista de letras.
        assert ConversationModel("Hola").answer_user() == "Hola"
        assert ParserModel({"a": 1}).parse_data() == {"a": 1}

    @pytest.mark.parametrize(
        "make", [lambda: ConversationModel([]), lambda: ParserModel([])], ids=["conv", "parser"]
    )
    def test_an_empty_script_is_rejected(self, make: Callable[[], object]) -> None:
        with pytest.raises(ValueError, match="al menos un valor"):
            make()

    def test_the_parser_script_cannot_be_changed_from_outside(self) -> None:
        script = {"a": 1}
        parser_model = ParserModel(script)
        script["a"] = 2  # quien lo creó modifica su dict...
        parser_model.parse_data()["a"] = 999  # ...y quien lo usa, el resultado

        assert parser_model.parse_data() == {"a": 1}
