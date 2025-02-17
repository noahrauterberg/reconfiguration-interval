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


# class GroundNetwork:
#     """
#     Simulation class for ground stations so that they can be simluated independently from the constellation
#     """
