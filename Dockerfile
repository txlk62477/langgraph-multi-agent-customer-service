FROM langchain/langgraph-api:3.12

RUN pip install --no-cache-dir 'playwright>=1.50,<2.0' \
    && python -m playwright install --with-deps chromium

ADD . /deps/agent

RUN cd /deps/agent \
    && PYTHONDONTWRITEBYTECODE=1 uv pip install --system --no-cache-dir \
       -c /api/constraints.txt -e .

ENV LANGGRAPH_CHECKPOINTER='{"path": "/deps/agent/src/agent/checkpointer.py:generate_checkpointer"}'
ENV LANGSERVE_GRAPHS='{"customer_service": "/deps/agent/src/agent/supervisor/graph.py:customer_service_graph", "general_qa_agent": "/deps/agent/src/agent/agents/general_qa.py:general_qa_agent", "rental_recommendation_agent": "/deps/agent/src/agent/agents/rental_recommendation.py:rental_recommendation_agent", "rental_booking_agent": "/deps/agent/src/agent/agents/rental_booking.py:rental_booking_agent", "order_history_agent": "/deps/agent/src/agent/agents/order_history.py:order_history_agent", "order_cancellation_agent": "/deps/agent/src/agent/agents/order_cancellation.py:order_cancellation_agent"}'

RUN mkdir -p /api/langgraph_api /api/langgraph_runtime /api/langgraph_license \
    && touch /api/langgraph_api/__init__.py /api/langgraph_runtime/__init__.py \
       /api/langgraph_license/__init__.py
RUN PYTHONDONTWRITEBYTECODE=1 uv pip install --system --no-cache-dir --no-deps -e /api

WORKDIR /deps/agent
