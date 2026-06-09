"""Version-agnostic shared plumbing for container implementations.

Code here is imported by individual version packages (v0_2, v0_3, ...). Versions
must never import from one another — shared logic lives here instead.
"""
