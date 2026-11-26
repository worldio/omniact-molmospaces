"""Procedural embodiment generation.

A generated embodiment is a sampled arm with one of MolmoSpaces' shipped grippers grafted
onto its flange, emitted as an ``EmbodimentSpec`` + MJCF pair that loads through the
ordinary :func:`molmo_spaces.robots.configured.loader.load_embodiment_spec` path.

Identity is a pure function of ``(seed, GENERATOR_VERSION)``, so a dataset row traces back
to the robot that produced it without shipping the robot alongside it.
"""
