# ODbL handling for OSM-derived Sionna scenes

**Project decision:** use the Sionna RT Etoile, Florence, and Munich scenes for research subject to ODbL compliance. This records the user's policy choice; it does not clear a particular dataset, figure, model artifact, or source-asset redistribution for release.

## Required attribution

Use the credit **“© OpenStreetMap contributors”** wherever an OSM-derived scene is used in a public figure, paper, or other produced work. Link the data notice to the OpenStreetMap copyright page. When distributing OSM-derived data, name and link the license directly as **Open Data Commons Open Database License (ODbL) v1.0**. Suggested figure credit:

> Scene geometry derived from OpenStreetMap data, © OpenStreetMap contributors, available under the Open Database License (ODbL) v1.0.

The [OpenStreetMap copyright and license notice](https://www.openstreetmap.org/copyright) requires contributor credit and a clear ODbL notice. The [ODbL v1.0 legal code](https://opendatacommons.org/licenses/odbl/1-0/) governs the database rights and share-alike terms. The [OSMF Produced Work guideline](https://osmfoundation.org/wiki/Licence/Community_Guidelines/Produced_Work_-_Guideline) distinguishes a work intended for extraction of source data from a produced work and says the underlying database or its alterations must be offered when publishing a produced work.

## Current scene bindings

The machine-readable companion [`odbl-osm-scene-release-policy-20261023.json`](odbl-osm-scene-release-policy-20261023.json) binds this policy to the three scene asset-tree hashes in the 2026-10-23 Sionna RT audit. That audit confirmed installed assets match the pinned Sionna checkout. It did not record the OSM extract timestamp, object/change-set identifiers, original conversion recipe, or a per-file third-party source/license closure; those fields remain unknown and must not be invented.

The scene geometry and associated OSM-derived data must be treated as ODbL data when redistributed. Sionna software's Apache-2.0 license is separate and does not replace the scene-data terms. For each planned release, classify the exact artifact set before distribution:

- If it conveys the OSM-derived database or a derivative database, provide it under ODbL v1.0 with the required notices and access.
- If it is a produced work, put attribution and a working ODbL link in a reasonable place for that medium, and make the underlying database or its alterations available as required by ODbL section 4.6.
- Do not assume that simulated path, channel, or model outputs are outside ODbL obligations. Their classification depends on the contents and intended use of the specific release and remains unresolved here.

## Release gate

The scenes are approved for internal research use under the selected policy. They are **not yet cleared for public dataset or asset redistribution**. Before any public release containing these scenes or outputs derived from them, the release record must identify:

1. The exact scene and asset-tree hashes, source software revision, and all additional source/license components.
2. The OSM extract/update date, source provenance, and the scene-conversion steps, or a documented upstream reason those records cannot be recovered.
3. The artifact-by-artifact ODbL classification, attribution placement, direct license link, and the database download/recreation offer where applicable.
4. Which files are software under Apache-2.0, OSM-derived data under ODbL, or separately licensed content, without combining them under one blanket license label.

The release validator implements this distinction in manifest schema `umcd.public_release.v2`, described structurally by `datasets/schema/public-release-manifest.v2.schema.json`: every artifact declares one ODbL classification for each source scene it uses. The validator rejects unversioned fields and requires canonical lowercase SHA-256 artifact digests. A scene-wide classification is rejected because a release may contain both a produced work and a database-derived artifact from the same scene. An artifact classified as an ODbL database must itself declare ODbL-1.0 and link directly to that license. Each assessed scene also needs an absolute HTTPS source-data offer URL. The exporter additionally requires a detached OpenSSH signature over the exact manifest bytes, verified against a separately supplied allowed-signers file under the `umcd-public-release-manifest-v2@github.com` namespace. The archive retains those exact manifest and signature bytes and a receipt validated by `datasets/schema/public-release-approval.v1.schema.json`. The signature authenticates a trusted principal for the declaration; it does not establish that the claims are true, that the URL is live, or that the legal classification is correct. The validator checks URL structure and manifest consistency only. These checks do not change the public-release gate above.

Until these fields are completed for the intended release, publish no OSM-derived scene bundle or derivative database. This gate is independent of the G3 physics qualification gate; passing one does not pass the other.
