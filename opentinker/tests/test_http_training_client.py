# Copyright 2025 OpenTinker
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

from opentinker.client.utils.http_training_client import (
    build_http_checkpoint_dir,
    resolve_http_checkpoint_dir,
)


def test_build_http_checkpoint_dir_uses_project_experiment_and_job_id():
    assert (
        build_http_checkpoint_dir("co-evolve", "sciworld_training", "5913f8d6")
        == "./ckpt/co-evolve/sciworld_training/job_5913f8d6"
    )


def test_build_http_checkpoint_dir_sanitizes_path_components():
    assert (
        build_http_checkpoint_dir("team alpha", "exp/one", "job:42")
        == "./ckpt/team_alpha/exp_one/job_job_42"
    )


def test_resolve_http_checkpoint_dir_prefers_explicit_override():
    args = SimpleNamespace(
        project_name="co-evolve",
        experiment_name="sciworld_training",
        ckpt_dir="./custom/ckpt/path",
        checkpoint_dir=None,
        job_id=None,
    )
    env = SimpleNamespace(job_id="5913f8d6")

    assert resolve_http_checkpoint_dir(args, env) == "./custom/ckpt/path"


def test_resolve_http_checkpoint_dir_falls_back_to_job_scoped_default():
    args = SimpleNamespace(
        project_name="co-evolve",
        experiment_name="sciworld_training",
        ckpt_dir=None,
        checkpoint_dir=None,
        job_id=None,
    )
    env = SimpleNamespace(job_id="5913f8d6")

    assert (
        resolve_http_checkpoint_dir(args, env)
        == "./ckpt/co-evolve/sciworld_training/job_5913f8d6"
    )
