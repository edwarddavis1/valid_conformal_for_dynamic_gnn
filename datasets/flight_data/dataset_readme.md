The data in this dataset is derived and cleaned from the full OpenSky
dataset. It spans all flights seen by the network's more than 2500
members between 1 January 2020 and 1 April 2020. More data will be
included in the dataset every month until the end of the COVID-19
pandemia.

## Disclaimer

The data provided in the files is provided as is. Some information could
be erroneous.

- Origin and destination airports are computed online based on
  trajectories in base: no crosschecking with external sources of data
  has been conducted. Fields **origin** or **destination** are empty
  when no airport could be found.
- Aircraft information come from the OpenSky aircraft database. Fields
  **typecode** and **registration** are empty when the aircraft is not
  present in the database.

## Description of the dataset

One file per month is provided as a csv file with the following
features:

- **callsign**: the identifier of the flight displayed on ATC screens
  (usually the first three letters are reserved for an airline: AFR
  for Air France, DLH for Lufthansa, etc.)
- **number**: the commercial number of the flight, when available (the
  matching with the callsign comes from public open API)
- **aircraft_uid**: a unique anonymised identifier for aircraft;
- **typecode**: the aircraft model type (when available);
- **origin**: a four letter code for the origin airport of the flight
  (when available);
- **destination**: a four letter code for the destination airport of
  the flight (when available);
- **firstseen**: the UTC timestamp of the first message received by
  the OpenSky Network;
- **lastseen**: the UTC timestamp of the last message received by the
  OpenSky Network;
- **day**: the UTC day of the last message received by the OpenSky
  Network.

## Examples

Possible visualisations of the data are available at the following
page:

<https://traffic-viz.github.io/scenarios/covid19.html>
