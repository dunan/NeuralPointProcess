DROP TABLE IF EXISTS badges_count;
CREATE TABLE badges_count AS (
    SELECT id, name, userid, date,
           rank() OVER (PARTITION BY userid, name ORDER BY date) AS count
    FROM badges
);
CREATE INDEX badges_count_date_idx   ON badges_count(date);
CREATE INDEX badges_count_userid_idx ON badges_count(userid);
CREATE INDEX badges_count_count_idx  ON badges_count(count);
CREATE INDEX badges_count_name_idx   ON badges_count(name);

-- Select the users who have earned more than
--   40 badges
-- in
--   '2012-01-01' and '2014-01-01'
--
-- Select only those badges which have been earned than more than
--   100 times
-- by these users in
--   '2012-01-01' and '2014-01-01'

DROP TABLE IF EXISTS badgeTypes;
CREATE TABLE badgeTypes AS
    SELECT row_number() OVER () AS Id,
           name AS badge
    FROM badges_count
    GROUP BY name
    ORDER BY name;
CREATE INDEX badge_type_id_idx ON badgeTypes(Id);
CREATE INDEX badge_type_badge_idx ON badgeTypes(badge);


DROP VIEW IF EXISTS relevant_users CASCADE;
CREATE VIEW relevant_users AS
    SELECT userid
    FROM badges_count BC1
    WHERE date BETWEEN '2012-01-01' AND '2014-01-01'
          AND name IN ('Favourite Question','Stellar Question','Nice Question','Good Question','Great Question','Popular Question','Notable Question','Famous Question','Enlightened','Guru','Nice Answer','Good Answer','Great Answer','Populist','Reversal','Revival','Necromancer','Activist','Campaigner','Founder','Good Question','Grassroots','Great Question','Movement','Nice Question','Promoter','Revolution','Caucus','Constituent','Yearling','Announcer','Booster','Publicist')
    GROUP BY userid
    HAVING count(*) > 40;


DROP VIEW IF EXISTS relevant_badges;
CREATE VIEW relevant_badges AS
    SELECT name
    FROM badges_count BC2
    WHERE EXISTS (SELECT * FROM relevant_users RU1 WHERE RU1.userid = BC2.userid)
          AND date BETWEEN '2012-01-01' AND '2014-01-01'
          -- These are the badges which can be awareded multiple times
          AND name IN ('Favourite Question','Stellar Question','Nice Question','Good Question','Great Question','Popular Question','Notable Question','Famous Question','Enlightened','Guru','Nice Answer','Good Answer','Great Answer','Populist','Reversal','Revival','Necromancer','Activist','Campaigner','Founder','Good Question','Grassroots','Great Question','Movement','Nice Question','Promoter','Revolution','Caucus','Constituent','Yearling','Announcer','Booster','Publicist')
    GROUP BY name
    HAVING count(*) > 100
    ORDER BY name;

-- EXPLAIN
-- SELECT count(*)               AS entries,
--        count(distinct userid) AS num_users,
--        count(distinct name)   AS num_badges
-- FROM badges_count BC3
-- WHERE EXISTS (SELECT * FROM relevant_badges RB1 WHERE RB1.name = BC3.name)
--     AND EXISTS (SELECT * FROM relevant_users RU2 WHERE RU2.userid = BC3.userid)
--     AND BC3.date BETWEEN '2012-01-01' AND '2014-01-01';


DROP TABLE IF EXISTS so_data;
CREATE TABLE so_data AS (
    SELECT userid,
           array_agg(BC3.name ORDER BY BC3.date)                     AS BadgeNames,
           array_agg(BT.id ORDER BY BC3.date)                        AS EventIds,
           array_agg(EXTRACT(EPOCH FROM BC3.date) ORDER BY BC3.date) AS TimeStamp,
           count(*)                                                  AS NumEvents
    FROM badges_count BC3
    JOIN badgeTypes BT ON (BT.badge = BC3.name)
    WHERE EXISTS (SELECT * FROM relevant_badges RB1 WHERE RB1.name = BC3.name)
        AND EXISTS (SELECT * FROM relevant_users RU2 WHERE RU2.userid = BC3.userid)
        AND BC3.date BETWEEN '2012-01-01' AND '2014-01-01'
    GROUP BY BC3.userid
);


