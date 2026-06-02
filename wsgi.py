"""WSGI entrypoint for production servers and Hugging Face Spaces.

Builds the app once on import and exposes that Flask instance as ``server`` so it
can be run with, e.g.::

    gunicorn wsgi:server --bind 0.0.0.0:7860

Importing this module triggers ``create_app()``, which loads the cached MMM
posterior (``data/mmm_idata.nc``) and registers every callback exactly as
``python app.py`` does for local development.
"""

from __future__ import annotations

from app import create_app

app = create_app()
server = app.server
