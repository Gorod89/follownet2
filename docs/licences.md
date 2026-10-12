# Data licences

What each raw data source allows, and what FollowNet-2 publishes for it. The statements were
checked on 2026-09-27 against the sources' own pages, APIs and licence files. The machine-readable
counterpart, with file lists, sizes and check sums, is `conf/sources.yaml`
(`python scripts/download_data.py --list`). This page records what the sources state; it is not
legal advice.

"Derived events" are the car-following events FollowNet-2 extracts: per-sample gap, speeds and
accelerations of a follower and its leader, i.e. trajectory data.

## Licence table

| Source | Licence | Licence URL | Access | Raw data: redistribution | Derived events: redistribution | FollowNet-2 publishes | Citation |
|---|---|---|---|---|---|---|---|
| FollowNet event sets (figshare) | CC0 1.0, on all 26 articles | https://creativecommons.org/publicdomain/zero/1.0/ | open, direct download | allowed | allowed | events, including the HighD-derived sets (note 1) | Chen et al. 2023 [1], for the HighD sets also [7] |
| OpenACC (JRC) | CC BY 4.0 | https://creativecommons.org/licenses/by/4.0/ | open, direct download | allowed with credit and changes indicated | allowed with credit and changes indicated | events | Makridis et al. 2021 [2] |
| NGSIM (data.transportation.gov) | two strings on the portal (note 3) | https://creativecommons.org/licenses/by-sa/4.0/ | open, Socrata API, no account | allowed, ShareAlike | allowed under CC BY-SA 4.0 only | events, as CC BY-SA 4.0 | US DOT FHWA [3] |
| Waymo car-following pairs, Hu et al. (Mendeley Data v3) | CC BY 4.0 | https://creativecommons.org/licenses/by/4.0/ | open, direct download (note 2) | allowed with credit | allowed with credit (note 2) | events | Hu et al. 2022 [4] |
| TGSIM (ITS DataHub) | US public domain | https://www.usa.gov/publicdomain/label/1.0/ | open, Socrata API, no account | allowed | allowed | events | US DOT FHWA [5] |
| UTE, Ubiquitous Traffic Eyes | none: no licence file; the README asks for a citation | https://github.com/Ruyi-Feng/Ubiquitous-Traffic-Eye | manual: Google Drive or Baidu links | unclear | unclear | code and check sums only | README [6] |
| highD (levelXdata) | non-commercial use; no redistribution | https://levelxdata.com/highd-dataset/ | by application | forbidden | forbidden | extraction code and event check sums, never the events | Krajewski et al. 2018 [7] |
| exiD (levelXdata) | non-commercial use; no redistribution | https://levelxdata.com/exid-dataset/ | by application | forbidden | forbidden | extraction code and event check sums, never the events | Moers et al. 2022 [8] |
| CitySim (UCF SST Lab) | terms of the signed request form: non-commercial research and teaching, no redistribution | https://github.com/UCF-SST-Lab/UCF-SST-CitySim1-Dataset | by signed form, sent by e-mail | forbidden | forbidden (note 6) | code and check sums only | Zheng et al. [9] |
| pNEUMA (EPFL) | ambiguous: CC BY-NC 4.0 or CC BY 4.0 (note 7) | https://open-traffic.epfl.ch/index.php/downloads/ | open: Zenodo, or the EPFL form with a terms tick-box | unclear | not published until EPFL clarifies | code and check sums only | Barmpounakis and Geroliminis 2020 [10] |

## Notes

1. **FollowNet.** figshare reports CC0 for every article, but the authors' notebook asks users not
   to distribute the data without their permission. CC0 is the legal grant; the request is a
   courtesy that the release notes should honour by pointing to figshare. The HighD and 30 s
   HighD sets are derived from highD, whose own terms forbid redistribution. FollowNet-2
   publishes the events it derives from these two figshare sets (decision of 2026-09-28), relying
   on the CC0 grant of the FollowNet authors, with the CC0 notice and the citations of FollowNet
   and highD. This does not extend to raw highD: events extracted from the levelXdata files are
   never published. The upstream terms of the Lyft and Waymo sets were not checked.
2. **Waymo pairs.** The Mendeley licence text warns that further permission may be needed for
   content of the dataset that belongs to a third party. The pairs are derived from the Waymo Open
   Dataset, whose own terms were not checked: check them before a release. On 2026-09-27
   data.mendeley.com refused python-requests with a browser challenge (HTTP 403). The script
   then asks for a manual download; it does not work around the challenge.
3. **NGSIM.** The portal's data set licence reads "Creative Commons Attribution | Share Alike 3.0
   Unported" (`CC_30_BY_SA`, repeated in the catalogue and in DCAT); its Common Core field
   `License` is `https://creativecommons.org/licenses/by-sa/4.0/`. Both are ShareAlike. CC BY-SA
   3.0 allows adaptations under a later version with the same elements, so derived events
   released under CC BY-SA 4.0 satisfy both readings. (cf-stability's `SOURCES.json` records
   "public domain (US DOT)", which contradicts the portal.)
4. **TGSIM.** The data set licence field is empty; the Common Core field `License` is the US public
   domain label `http://www.usa.gov/publicdomain/label/1.0/`. No licence condition applies, but each
   data set page gives a recommended citation.
5. **UTE.** No licence is stated anywhere (repository, READMEs, Drive folders), and the READMEs
   call the release sample data. Being publicly reachable does not grant rights, so nothing derived
   from it is published until the authors grant a licence.
6. **CitySim.** The GitHub repository is Apache-2.0, but that covers the repository contents only.
   The data fall under the request form, which allows derivative works only if the data cannot be
   recovered from them. Events contain the trajectories, so they are not publishable. The
   collaboration sites (Southwest Jiaotong University, Southeast University, PolyU) are
   distributed by those teams.
7. **pNEUMA.** The download page `https://open-traffic.epfl.ch/index.php/downloads/` makes two
   statements, quoted literally:
   - under the DOWNLOAD button: "pNEUMA dataset © 2020 by E. Barmpounakis and N. Geroliminis is
     licensed under CC BY-NC 4.0" (linked to `http://creativecommons.org/licenses/by-nc/4.0/?ref=chooser-v1`);
   - in the FAQ "How can I use the pNEUMA and pNEUMA Vision dataset?": "pNEUMA dataset © 2020
     by E. Barmpounakis and N. Geroliminis is licensed under CC BY 4.0" (linked to
     `https://creativecommons.org/licenses/by/4.0/legalcode`).

   Zenodo (`10.5281/zenodo.10491409`) says `cc-by-4.0`. The terms and conditions page names no CC
   licence; it requires crediting the article and the Zenodo DOI, and the acknowledgement "Data
   source: pNEUMA – open-traffic.epfl.ch". Until EPFL clarifies, FollowNet-2 assumes the stricter
   CC BY-NC 4.0 and publishes no data derived from pNEUMA.
8. **exiD.** The format document says the map-based columns (`leadId`, `leadDHW`, `leadDV`,
   `laneletId`, ...) are part of `XX_tracks.csv` only if they belong to the purchased package.
   Whether the academic release has them was not checked.

## What this means for a benchmark release

- **A release has no single licence.** Events are published per source, in separate files, each
  with its own licence and citation: CC0 (FollowNet), CC BY 4.0 (OpenACC, Waymo pairs), CC BY-SA
  4.0 (NGSIM) and public domain (TGSIM). NGSIM's ShareAlike rules out merging its events into
  a file under another licence.
- **Credit and changes.** The CC BY and BY-SA sources require stating what was changed. The
  release notes describe the processing: extraction criteria, resampling to 10 Hz and
  reconstruction, with the config hash.
- **Restricted sources are reproducible, not redistributed.** For highD, exiD, CitySim, UTE
  and pNEUMA the release contains the extraction code, the
  exact configuration, the sha256 of every raw file (from `SOURCES.json`) and the sha256 of every
  derived event file. A user with their own licence rebuilds the events and checks them bit for bit.
- **Results must not leak the data.** For restricted sources, publish aggregate and per-event
  metrics only. Per-sample outputs, such as simulated trajectories driven by the recorded leader,
  contain the recorded data.
- **No raw data in the release or in git.** `data/raw/` stays in `.gitignore`; the download script
  and this page tell users where to get each source.
- **Before a release,** close the open items: the Waymo Open Dataset terms (note 2), the upstream
  terms of FollowNet's Lyft and Waymo sets (note 1), EPFL's answer on pNEUMA (note 7) and a
  licence from the UTE authors (note 5).

## References

References marked \* are given from the literature and were not re-checked against the
publisher on 2026-09-27.

1. Chen, X., Zhu, M., Chen, K. et al. FollowNet: A Comprehensive Benchmark for Car-Following
   Behavior Modeling. Sci Data 10, 828 (2023). https://doi.org/10.1038/s41597-023-02718-7. Data:
   Zhu, M., Chen, X., Chen, K., Wang, P., Lu, H., Zhong, H., et al. (2023). figshare.
   https://doi.org/10.6084/m9.figshare.c.6777810.v1
2. \* Makridis, M., Mattas, K., Anesiadou, A., Ciuffo, B. (2021). OpenACC. An open database of
   car-following experiments to study the properties of commercial ACC systems. Transportation
   Research Part C 125, 103047. https://doi.org/10.1016/j.trc.2021.103047. Data © European Union,
   licensed under CC BY 4.0 (`copyright.txt` of every campaign).
3. U.S. Department of Transportation Federal Highway Administration. Next Generation Simulation
   (NGSIM) Vehicle Trajectories and Supporting Data [Dataset]. Provided by ITS DataHub through
   Data.transportation.gov. https://doi.org/10.21949/1504477
4. \* Hu, X., Zheng, Z., Chen, D., Zhang, X., Sun, J. (2022). Processing, assessing, and enhancing
   the Waymo autonomous vehicle open dataset for driving behavior research. Transportation
   Research Part C 134, 103490. https://doi.org/10.1016/j.trc.2021.103490. Data: Mendeley Data, V3.
   https://doi.org/10.17632/wfn2c3437n.3
5. U.S. Department of Transportation Federal Highway Administration (2023, 2024). Third Generation
   Simulation Data (TGSIM) <data set> Trajectories [Dataset]. Provided by ITS DataHub through
   Data.transportation.gov. One entry per data set used, with the DOI given on its page.
6. Ubiquitous Traffic Eyes, https://github.com/Ruyi-Feng/Ubiquitous-Traffic-Eye: cite what the
   repository README asks for.
7. \* Krajewski, R., Bock, J., Kloeker, L., Eckstein, L. (2018). The highD Dataset: A Drone Dataset
   of Naturalistic Vehicle Trajectories on German Highways for Validation of Highly Automated
   Driving Systems. 21st International Conference on Intelligent Transportation Systems (ITSC),
   2118-2125. https://doi.org/10.1109/ITSC.2018.8569552
8. \* Moers, T., Vater, L., Krajewski, R., Bock, J., Zlocki, A., Eckstein, L. (2022). The exiD
   Dataset: A Real-World Trajectory Dataset of Highly Interactive Highway Scenarios in Germany.
   IEEE Intelligent Vehicles Symposium (IV), 958-964. https://doi.org/10.1109/IV51971.2022.9827305
9. \* Zheng, O., Abdel-Aty, M., Yue, L., Abdelraouf, A., Wang, Z., Mahmoud, N. CitySim: A
   Drone-Based Vehicle Trajectory Dataset for Safety-Oriented Research and Digital Twins.
   Transportation Research Record. https://doi.org/10.1177/03611981231185768
10. Barmpounakis, E., Geroliminis, N. (2020). On the new era of urban traffic monitoring with
    massive drone data: The pNEUMA large-scale field experiment. Transportation Research Part C
    111, 50-71. https://doi.org/10.1016/j.trc.2019.11.023. Data: https://doi.org/10.5281/zenodo.10491409.
    Acknowledgement: "Data source: pNEUMA – open-traffic.epfl.ch".
