import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import SecretStr

from repopilot.config import Settings
from repopilot.model_routing import ModelProviderRegistry
from repopilot.models import (
    AgentModelRoute,
    ModelRoutingConfig,
    ProviderCredentialCreate,
)


class _FakeModels:
    def list(self):
        return SimpleNamespace(
            data=[
                SimpleNamespace(
                    id="third-party-code",
                    owned_by="example-provider",
                ),
                SimpleNamespace(
                    id="third-party-reasoning",
                    owned_by="example-provider",
                ),
            ]
        )


class _FakeOpenAI:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.models = _FakeModels()
        self.closed = False

    def close(self):
        self.closed = True


class ModelRoutingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.settings = Settings(
            deepseek_api_key=None,
            deepseek_base_url="https://api.deepseek.com",
            deepseek_model="deepseek-v4-pro",
            llm_mode="mock",
            max_files=10,
            max_file_bytes=10_000,
            rag_top_k=3,
            test_timeout_seconds=10,
            runtime_dir=Path(self.temp.name),
        )

    def tearDown(self):
        self.temp.cleanup()

    @patch(
        "repopilot.model_routing.socket.getaddrinfo",
        return_value=[
            (2, 1, 6, "", ("104.21.1.20", 443)),
        ],
    )
    def test_zzshu_is_a_supported_public_compatible_endpoint(self, _):
        registry = ModelProviderRegistry(self.settings)
        self.assertEqual(
            registry.validate_base_url("https://zzshu.cc/v1/"),
            "https://zzshu.cc/v1",
        )
        zzshu = next(
            item
            for item in registry.catalog()
            if item["preset_id"] == "zzshu"
        )
        self.assertEqual(zzshu["base_url"], "https://zzshu.cc/v1")

    @patch(
        "repopilot.model_routing.socket.getaddrinfo",
        return_value=[
            (2, 1, 6, "", ("104.21.1.20", 443)),
        ],
    )
    @patch("repopilot.model_routing.OpenAI", _FakeOpenAI)
    def test_byok_secret_stays_out_of_route_and_agents_resolve_independently(
        self, _
    ):
        registry = ModelProviderRegistry(self.settings)
        credential = registry.add_credential(
            ProviderCredentialCreate(
                preset_id="zzshu",
                display_name="My third-party gateway",
                base_url="https://zzshu.cc/v1",
                api_key=SecretStr("sk-super-secret-value"),
            )
        )
        routing = ModelRoutingConfig(
            preset="cross_review",
            default_route=AgentModelRoute(
                credential_id=credential.credential_id,
                model="third-party-code",
            ),
            agent_routes={
                "DevelopmentTeamAgent": AgentModelRoute(
                    credential_id=credential.credential_id,
                    model="third-party-code",
                ),
                "ReleaseReviewerAgent": AgentModelRoute(
                    credential_id=credential.credential_id,
                    model="third-party-reasoning",
                ),
            },
        )
        validation = registry.validate_routing(routing)
        self.assertTrue(validation.valid)
        developer = registry.resolve(routing, "generated_project")
        reviewer = registry.resolve(routing, "studio_review")
        self.assertEqual(developer["agent_id"], "DevelopmentTeamAgent")
        self.assertEqual(developer["model"], "third-party-code")
        self.assertEqual(reviewer["agent_id"], "ReleaseReviewerAgent")
        self.assertEqual(reviewer["model"], "third-party-reasoning")
        self.assertNotIn("sk-super-secret", routing.model_dump_json())
        self.assertEqual(
            registry.redact("failure sk-super-secret-value"),
            "failure [REDACTED]",
        )

    @patch(
        "repopilot.model_routing.socket.getaddrinfo",
        return_value=[
            (2, 1, 6, "", ("127.0.0.1", 11434)),
        ],
    )
    def test_private_model_endpoints_are_disabled_by_default(self, _):
        registry = ModelProviderRegistry(self.settings)
        with self.assertRaisesRegex(ValueError, "私网/本机"):
            registry.validate_base_url("http://127.0.0.1:11434/v1")


if __name__ == "__main__":
    unittest.main()
