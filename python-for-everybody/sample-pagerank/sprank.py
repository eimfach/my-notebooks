import sqlite3

conn = sqlite3.connect('spider.sqlite')
cursor = conn.cursor()

# Find the ids that send out page rank - we only are interested
# in pages in the SCC that have in and out links
# cache all distinct from_ids
cursor.execute('''SELECT DISTINCT from_id FROM Links''')
from_ids = list()
for row in cursor: 
    from_ids.append(row[0])

# Find the ids that receive page rank 
# These Lists are for caching
to_ids = list()
links = list()
# get all links but filter out the ones which are equal
cursor.execute('''SELECT DISTINCT from_id, to_id FROM Links''')
for row in cursor:
    from_id = row[0]
    to_id = row[1]
    # drop links where a page points to itself
    if from_id == to_id : continue
    # drop links that are pointing nowhere (Note: I guess this covers manual made entries)
    if from_id not in from_ids : continue
    # drop pages which weren't retrieved yet or point nowhere
    if to_id not in from_ids : continue
    links.append(row)
    # keep track of the entire superset of to_ids, the destination IDs
    # this is getting what's called the strongly connected component.
    # meaning that any of these IDs, there is a path from every (page) ID to every other (page) ID eventually.
    # that's called the strongly connected component in graph theory
    if to_id not in to_ids : to_ids.append(to_id)

# Get latest page ranks for strongly connected component
prev_ranks = dict()
for node in from_ids:
    cursor.execute('''SELECT new_rank FROM Pages WHERE id = ?''', (node, ))
    row = cursor.fetchone()
    prev_ranks[node] = row[0]

sval = input('How many iterations:')
many = 1
if ( len(sval) > 0 ) : many = int(sval)

# Sanity check
if len(prev_ranks) < 1 : 
    print("Nothing to page rank.  Check data.")
    quit()

# Lets do Page Rank in memory so it is really fast
for i in range(many):
    # print prev_ranks.items()[:5]
    next_ranks = dict();
    total = 0.0
    # Note: Why use list() on dict.items() ?
    for (node, old_rank) in list(prev_ranks.items()):
        total = total + old_rank
        next_ranks[node] = 0.0
    # print total

    # Find the number of outbound links and sent the page rank down each
    for (node, old_rank) in list(prev_ranks.items()):
        # print node, old_rank
        give_ids = list()
        for (from_id, to_id) in links:
            if from_id != node : continue
           #  print '   ',from_id,to_id

            if to_id not in to_ids: continue
            give_ids.append(to_id)
        if ( len(give_ids) < 1 ) : continue
        amount = old_rank / len(give_ids)
        # print node, old_rank,amount, give_ids
    
        for id in give_ids:
            next_ranks[id] = next_ranks[id] + amount
    
    newtot = 0
    for (node, next_rank) in list(next_ranks.items()):
        newtot = newtot + next_rank
    evap = (total - newtot) / len(next_ranks)

    # print newtot, evap
    for node in next_ranks:
        next_ranks[node] = next_ranks[node] + evap

    newtot = 0
    for (node, next_rank) in list(next_ranks.items()):
        newtot = newtot + next_rank

    # Compute the per-page average change from old rank to new rank
    # As indication of convergence of the algorithm
    totdiff = 0
    for (node, old_rank) in list(prev_ranks.items()):
        new_rank = next_ranks[node]
        diff = abs(old_rank-new_rank)
        totdiff = totdiff + diff

    avediff = totdiff / len(prev_ranks)
    print(i+1, avediff)

    # rotate
    prev_ranks = next_ranks

# Put the final ranks back into the database

print(list(next_ranks.items())[:5])
cursor.execute('''UPDATE Pages SET old_rank=new_rank''')
for (id, new_rank) in list(next_ranks.items()) :
    cursor.execute('''UPDATE Pages SET new_rank=? WHERE id=?''', (new_rank, id))
conn.commit()
cursor.close()

