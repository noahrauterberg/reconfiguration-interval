#
# This file is part of reconfiguration-interval
# (https://github.com/noahrauterberg/reconfiguration-interval).
# Copyright (c) 2025 Noah Rauterberg.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#


class GroundStation:
    """
    Ground station configuration
    """

    def __init__(
        self,
        name: str,
        lat: float,
        lng: float,
        min_elevation: float,
    ):
        """
        Ground station configuration.

        :param name: The name of the ground station.
        :param lat: The latitude of the ground station.
        :param lng: The longitude of the ground station.
        :param min_elevation: The minimum elevation of the ground station in degrees.
        """

        self.name = name
        self.lat = lat
        self.lng = lng
        self.min_elevation = min_elevation
