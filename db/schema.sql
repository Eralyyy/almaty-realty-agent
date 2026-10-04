CREATE TABLE IF NOT EXISTS listings (
    id            BIGINT PRIMARY KEY,          -- krisha.kz advert id
    url           TEXT NOT NULL,
    title         TEXT NOT NULL,               -- "3-комнатная квартира · 56 м² · 1/4 этаж"
    rooms         SMALLINT,
    area_m2       REAL,
    floor         SMALLINT,
    floors_total  SMALLINT,
    price_kzt     BIGINT,
    district      TEXT,                        -- "Бостандыкский р-н"
    address       TEXT,                        -- "мкр Коктем-1"
    description   TEXT,                        -- short preview text from the card
    house_type    TEXT,                        -- "панельный", "кирпичный", "монолитный", ...
    year_built    SMALLINT,
    scraped_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS listings_district_idx ON listings (district);
CREATE INDEX IF NOT EXISTS listings_price_idx    ON listings (price_kzt);
CREATE INDEX IF NOT EXISTS listings_rooms_idx    ON listings (rooms);
