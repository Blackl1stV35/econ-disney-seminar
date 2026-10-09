# Investment figures converted to US dollars

The file data/processed/cases_investment_fx.csv holds 27 rows. Its dollar costs are derived estimates built from reported local-currency figures and official annual-average exchange rates. They are not reported costs.

Twenty rows were converted with the World Bank indicator PA.NUS.FCRF for the price year, and seven rows were already in dollars and carry a rate of 1. Every rate was fetched twice from different World Bank addresses and agreed in full, so no rate is unverified and no converted cost is blank. Fifteen rows have price_year_assumed equal to Y, meaning no price year was stated and the opening year was used. Where a local-currency figure existed it was preferred over dollar figures of unknown conversion. For Niagara Fallsview the sources print only a dollar sign, and Canadian dollars were assumed. The column n_figures_considered counts the in-scope figures weighed under the rule applied.

The figures conflict most for VNM_2013_grand_ho_tram, CHN_2014_chimelong_ocean_kingdom, TUR_2016_land_of_legends, USA_2001_disney_california_adventure and MYS_2012_legoland_malaysia, mostly because the sources describe different scopes. Among figures pooled in a median, ESP_2000_terra_mitica differs most, from EUR270m to EUR421m.

Seven catalogue rows are not in the file. SGP_2010_universal_studios_singapore, PHL_2009_resorts_world_manila, DEU_2002_legoland_deutschland and NLD_2001_toverland have no figure. ARE_2010_ferrari_world_abu_dhabi, USA_2011_legoland_florida and DEU_2004_tropical_islands have only figures that are not the opening cost.
