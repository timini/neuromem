"""Unit tests for neuromem.storage.base — ABC instantiation behaviour.

Tests the StorageAdapter ABC's refusal to instantiate when abstract
methods are missing. The 14 behavioural contract tests (insert round-
trip, decay+archive, etc.) live in test_storage_adapter_contract.py
and are parameterised over concrete adapters.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from neuromem.storage.base import StorageAdapter, StorageError
from numpy.typing import NDArray


class TestStorageError:
    def test_is_runtime_error_subclass(self) -> None:
        assert issubclass(StorageError, RuntimeError)

    def test_can_be_raised_and_caught(self) -> None:
        with pytest.raises(StorageError, match="boom"):
            raise StorageError("boom")


class TestStorageAdapterABC:
    def test_direct_instantiation_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            StorageAdapter()  # type: ignore[abstract]

    def test_empty_subclass_cannot_instantiate(self) -> None:
        class Empty(StorageAdapter):
            pass

        with pytest.raises(TypeError, match="abstract"):
            Empty()  # type: ignore[abstract]

    def test_subclass_missing_one_method_cannot_instantiate(self) -> None:
        """Implement 13 of 14 methods; instantiation still fails."""

        class Incomplete(StorageAdapter):
            def insert_memory(
                self,
                raw_content: str,
                summary: str,
                metadata: dict[str, Any] | None = None,
            ) -> str:
                return "mem_1"

            def count_memories_by_status(self, status: str) -> int:
                return 0

            def get_memories_by_status(
                self,
                status: str,
                limit: int | None = None,
            ) -> list[dict[str, Any]]:
                return []

            def update_memory_status(
                self,
                memory_ids: list[str],
                new_status: str,
            ) -> None:
                return None

            def get_memory_by_id(self, memory_id: str) -> dict[str, Any] | None:
                return None

            def upsert_node(
                self,
                node_id: str,
                label: str,
                embedding: NDArray[np.floating] | list[float],
                is_centroid: bool,
            ) -> None:
                return None

            def update_node_labels(self, updates: dict[str, str]) -> None:
                return None

            def update_junction_summaries(self, updates: dict[str, str]) -> None:
                return None

            def get_all_nodes(self) -> list[dict[str, Any]]:
                return []

            def insert_edge(
                self,
                source_id: str,
                target_id: str,
                weight: float,
                relationship: str,
            ) -> None:
                return None

            def remove_edges_for_memory(self, memory_id: str) -> None:
                return None

            def get_nearest_nodes(
                self,
                query_embedding: NDArray[np.floating] | list[float],
                top_k: int = 5,
            ) -> list[dict[str, Any]]:
                return []

            def get_subgraph(
                self,
                root_node_ids: list[str],
                depth: int = 2,
            ) -> dict[str, Any]:
                return {"nodes": [], "edges": [], "memories": []}

            def apply_decay_and_archive(
                self,
                decay_lambda: float,
                archive_threshold: float,
                current_timestamp: int,
            ) -> list[str]:
                return []

            # spike_access_weight deliberately missing.

        with pytest.raises(TypeError, match="abstract"):
            Incomplete()  # type: ignore[abstract]

    def test_fully_stubbed_subclass_instantiates(self) -> None:
        """Implement all 14 methods; instantiation succeeds."""

        class Full(StorageAdapter):
            def insert_memory(
                self,
                raw_content: str,
                summary: str,
                metadata: dict[str, Any] | None = None,
            ) -> str:
                return "mem_1"

            def count_memories_by_status(self, status: str) -> int:
                return 0

            def get_memories_by_status(
                self,
                status: str,
                limit: int | None = None,
            ) -> list[dict[str, Any]]:
                return []

            def update_memory_status(
                self,
                memory_ids: list[str],
                new_status: str,
            ) -> None:
                return None

            def get_memory_by_id(self, memory_id: str) -> dict[str, Any] | None:
                return None

            def upsert_node(
                self,
                node_id: str,
                label: str,
                embedding: NDArray[np.floating] | list[float],
                is_centroid: bool,
            ) -> None:
                return None

            def update_node_labels(self, updates: dict[str, str]) -> None:
                return None

            def update_junction_summaries(self, updates: dict[str, str]) -> None:
                return None

            def get_all_nodes(self) -> list[dict[str, Any]]:
                return []

            def insert_edge(
                self,
                source_id: str,
                target_id: str,
                weight: float,
                relationship: str,
            ) -> None:
                return None

            def remove_edges_for_memory(self, memory_id: str) -> None:
                return None

            def get_nearest_nodes(
                self,
                query_embedding: NDArray[np.floating] | list[float],
                top_k: int = 5,
            ) -> list[dict[str, Any]]:
                return []

            def get_subgraph(
                self,
                root_node_ids: list[str],
                depth: int = 2,
            ) -> dict[str, Any]:
                return {"nodes": [], "edges": [], "memories": []}

            def apply_decay_and_archive(
                self,
                decay_lambda: float,
                archive_threshold: float,
                current_timestamp: int,
            ) -> list[str]:
                return []

            def spike_access_weight(
                self,
                memory_ids: list[str],
                timestamp: int,
            ) -> None:
                return None

            def set_summaries(self, updates: dict[str, str]) -> None:
                return None

        instance = Full()
        assert instance.insert_memory("raw", "summary") == "mem_1"
        assert instance.count_memories_by_status("inbox") == 0
        assert instance.get_all_nodes() == []

    def test_missing_set_summaries_cannot_instantiate(self) -> None:
        """set_summaries is abstract (ADR-004 contract). Omitting it
        must fail at instantiation so a silent data-loss bug can't
        ship. Same invariant as ``test_subclass_missing_one_method``
        but specifically for the post-ADR-004 method that the PR
        #59 review caught."""

        class NoSetSummaries(StorageAdapter):
            def insert_memory(
                self,
                raw_content: str,
                summary: str,
                metadata: dict[str, Any] | None = None,
            ) -> str:
                return "m"

            def count_memories_by_status(self, status: str) -> int:
                return 0

            def get_memories_by_status(
                self, status: str, limit: int | None = None
            ) -> list[dict[str, Any]]:
                return []

            def update_memory_status(self, memory_ids: list[str], new_status: str) -> None:
                return

            def get_memory_by_id(self, memory_id: str) -> dict[str, Any] | None:
                return None

            def upsert_node(
                self,
                node_id: str,
                label: str,
                embedding: NDArray[np.floating] | list[float],
                is_centroid: bool,
            ) -> None:
                return

            def update_node_labels(self, updates: dict[str, str]) -> None:
                return

            def update_junction_summaries(self, updates: dict[str, str]) -> None:
                return

            def get_all_nodes(self) -> list[dict[str, Any]]:
                return []

            def insert_edge(
                self,
                source_id: str,
                target_id: str,
                weight: float,
                relationship: str,
            ) -> None:
                return

            def remove_edges_for_memory(self, memory_id: str) -> None:
                return

            def get_nearest_nodes(
                self,
                query_embedding: NDArray[np.floating] | list[float],
                top_k: int = 5,
            ) -> list[dict[str, Any]]:
                return []

            def get_subgraph(self, root_node_ids: list[str], depth: int = 2) -> dict[str, Any]:
                return {"nodes": [], "edges": [], "memories": []}

            def apply_decay_and_archive(
                self,
                decay_lambda: float,
                archive_threshold: float,
                current_timestamp: int,
            ) -> list[str]:
                return []

            def spike_access_weight(self, memory_ids: list[str], timestamp: int) -> None:
                return

            # set_summaries deliberately missing.

        with pytest.raises(TypeError, match="abstract"):
            NoSetSummaries()  # type: ignore[abstract]
