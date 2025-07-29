#define USE_ROOT_GROUP 0 // If we use root_group for synchonization betweeen horiz and vert

// Wheter to use full sync or partial sync (From experiments partiall sync requires too many registers to work)
#define FULL_SYNC 1

#define MULTI_ROW_WG 0 // NOT IN USE

#define MINIMAL_WINDOW 0

//  -> If true window is span * 2 otherwise it's span * 2 + 2
//  -> Minimal is different from popsift but seems like filter
//  -> is always 0 for filter[span] hence removing that
//  -> compute and memory need

#define ASYNC_WRITE 0 // Async write both intermediate and DoG so need two per so like adding 4 rows
