"""Executor - performs a single bounded subtask using LLM."""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from desman.core.models import Subtask

if TYPE_CHECKING:
    from desman.router import ModelRouter


class Executor:
    """Executes a single subtask and produces artifacts using LLM."""

    def __init__(
        self,
        executor_id: str,
        workspace_root: Path,
        model_router: ModelRouter | None = None,
    ) -> None:
        self.executor_id = executor_id
        self.workspace_root = workspace_root
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.model_router = model_router

    def _build_execution_prompt(self, subtask: Subtask, checkpoint: dict[str, Any] | None) -> str:
        """Build prompt for LLM execution."""
        inputs = json.dumps(subtask.inputs_json, ensure_ascii=False, indent=2)
        deps = json.dumps(subtask.dependencies_json, ensure_ascii=False, indent=2)
        checkpoint_str = json.dumps(checkpoint, ensure_ascii=False, indent=2) if checkpoint else "none"

        return f"""You are an AI agent executing a specific subtask. Complete the work and return the result.

**Subtask:** {subtask.description}
**Acceptance Criteria:** {subtask.acceptance_criteria}

**Inputs (JSON):**
{inputs}

**Dependencies (completed subtask IDs):**
{deps}

**Previous Checkpoint:**
{checkpoint_str}

**Instructions:**
1. Do the work described in the subtask
2. Use the inputs and any context from dependencies
3. If you have a checkpoint, continue from where you left off
4. Return ONLY a JSON object with your result:
   {{
     "output": "your result text, code, findings, etc.",
     "files": {{"filename.py": "file content"}},  // optional files created
     "next_checkpoint": {{"step": N, "data": "..."}}  // for resumability
   }}

Do not include explanations outside the JSON."""

    async def execute(self, subtask: Subtask, checkpoint: dict[str, Any] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Execute a subtask using LLM. Returns (artifacts, new_checkpoint)."""
        if self.model_router is None:
            # Fallback: placeholder artifact
            fallback_artifact = {
                "type": "result",
                "content": {
                    "description": subtask.description,
                    "status": "completed",
                    "note": "Executor placeholder - no model router configured",
                },
                "path": None,
            }
            new_checkpoint = {
                "step": (checkpoint or {}).get("step", 0) + 1,
                "completed_at": time.time(),
                "subtask_id": str(subtask.id),
            }
            return [fallback_artifact], new_checkpoint

        # Build prompt and call LLM
        prompt = self._build_execution_prompt(subtask, checkpoint)

        response = ""
        try:
            response = await self.model_router.complete_simple(
                system_prompt="You are a precise AI executor. Return only valid JSON.",
                user_prompt=prompt,
                task_type="execution",
            )

            # Parse LLM response
            result = json.loads(response)

            # Write any files to workspace
            files_created = []
            for filename, content in result.get("files", {}).items():
                filepath = self.write_file(filename, content)
                files_created.append(str(filepath))

            # Main artifact
            success_artifact: dict[str, Any] = {
                "type": "result",
                "content": {
                    "description": subtask.description,
                    "output": result.get("output", ""),
                    "files_created": files_created,
                },
                "path": files_created[0] if files_created else None,
            }

            # Checkpoint for resumability
            next_checkpoint = result.get("next_checkpoint", {})
            new_checkpoint = {
                "step": (checkpoint or {}).get("step", 0) + 1,
                "completed_at": time.time(),
                "subtask_id": str(subtask.id),
                "llm_output": result.get("output", "")[:500],  # Truncated for storage
                **next_checkpoint,
            }

            return [success_artifact], new_checkpoint

        except json.JSONDecodeError as e:
            # LLM returned invalid JSON
            raw_resp = response[:1000] if 'response' in locals() else "no response"
            error_artifact = {
                "type": "error",
                "content": {
                    "description": subtask.description,
                    "error": f"Invalid JSON from LLM: {e}",
                    "raw_response": raw_resp,
                },
                "path": None,
            }
            new_checkpoint = {
                "step": (checkpoint or {}).get("step", 0) + 1,
                "completed_at": time.time(),
                "subtask_id": str(subtask.id),
                "error": str(e),
            }
            return [error_artifact], new_checkpoint

        except Exception as e:
            # Any other error
            error_artifact = {
                "type": "error",
                "content": {
                    "description": subtask.description,
                    "error": str(e),
                },
                "path": None,
            }
            new_checkpoint = {
                "step": (checkpoint or {}).get("step", 0) + 1,
                "completed_at": time.time(),
                "subtask_id": str(subtask.id),
                "error": str(e),
            }
            return [error_artifact], new_checkpoint

    def run_code_sandboxed(self, code: str, timeout: int = 30) -> tuple[str, str, int]:
        """Run Python code in a sandboxed subprocess. Returns (stdout, stderr, returncode)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            script_path = f.name

        try:
            result = subprocess.run(
                ["python3", script_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.workspace_root,
            )
            return result.stdout, result.stderr, result.returncode
        except subprocess.TimeoutExpired:
            return "", f"Timeout after {timeout}s", -1
        finally:
            Path(script_path).unlink(missing_ok=True)

    def write_file(self, path: str, content: str) -> Path:
        """Write a file within workspace."""
        full_path = self.workspace_root / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)
        return full_path

    def read_file(self, path: str) -> str:
        """Read a file from workspace."""
        return (self.workspace_root / path).read_text()

    def list_files(self, pattern: str = "**/*") -> list[Path]:
        """List files in workspace."""
        return list(self.workspace_root.glob(pattern))
