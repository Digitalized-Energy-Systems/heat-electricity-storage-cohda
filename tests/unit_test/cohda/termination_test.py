
import fractions
from typing import Any, Dict
import uuid
from unittest import mock
from mango.role.core import  RoleAgent
from mango.messages.codecs import JSON
from mango.role.api import SimpleReactiveRole
import pytest
from mango_library.coalition.core import CoalitionAssignment, CoalitionModel
from mango_library.negotiation.core import Negotiation, NegotiationModel
from mango_library.negotiation.termination import NegotiationTerminationRole, TerminationMessage
from mango_library.negotiation.util import extra_serializers


def test_serialization():
    codec = JSON()
    for serializer in extra_serializers:
        codec.add_serializer(*serializer())
    my_data = TerminationMessage(coalition_id=uuid.uuid1(), negotiation_id=uuid.uuid1(), weight=fractions.Fraction(1, 2))

    encoded = codec.encode(my_data)
    decoded = codec.decode(encoded)

    assert my_data.coalition_id == decoded.coalition_id
    assert my_data.negotiation_id == decoded.negotiation_id
    assert my_data.weight == decoded.weight
