"""Compatibility note for local adapter placement.

The runnable adapter lives at ``train.verl_agent_loop_adapter`` because the
cloned upstream verl package is a regular Python package named ``verl`` and
would otherwise shadow this project's ``verl/adapters`` directory.
"""
