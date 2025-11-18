from app import SessionLocal, create_db_and_seed, rag_pipeline


def test_retrieve_context_returns_ranked_sources():
    import asyncio

    asyncio.run(create_db_and_seed())
    with SessionLocal() as session:
        context, sources = rag_pipeline.retrieve_context("SQL", session)
    assert context
    assert len(sources) > 0
    assert sorted(sources, key=lambda s: s.score, reverse=True)[0].score >= sources[-1].score
