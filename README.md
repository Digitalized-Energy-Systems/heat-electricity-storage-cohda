# mango-library

The project is made for 'mango-agents==0.4.0'.
For faster calculation the **line 278** of **site-packages/mango/core/agent.py** has to be changed.
It is running **<method 'format' of 'str' objects>**, but the messages are large.
To avoid this remove **str(message)** out of:
```
logger.debug('Agent %s: Received message;%s}', self.aid, str(message))
```
