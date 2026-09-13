"""
Product-level services built on top of src/data/basketball's provider
boundary and the existing frozen prediction model.

This package is where "predict one player" and "build today's board of
predictions" live as plain, Streamlit-free functions -- see
prediction_service.py. Nothing here retrains, re-tunes, or otherwise
modifies models/points_regression.pkl or models/registry/CURRENT_V1;
both remain fixed prediction targets, not development targets, from
this package's point of view.
"""
