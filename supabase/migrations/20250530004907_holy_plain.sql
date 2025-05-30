-- Initialize pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create a new schema for EchoVault data
CREATE SCHEMA IF NOT EXISTS echovault;

-- Create a new role for EchoVault service
DO
$do$
BEGIN
   IF NOT EXISTS (
      SELECT FROM pg_catalog.pg_roles
      WHERE  rolname = 'echovault_rw') THEN
      CREATE ROLE echovault_rw LOGIN PASSWORD 'echovault';
   END IF;
END
$do$;

-- Grant permissions to the EchoVault role
GRANT USAGE ON SCHEMA public TO echovault_rw;
GRANT USAGE ON SCHEMA echovault TO echovault_rw;

-- Set default privileges
ALTER DEFAULT PRIVILEGES IN SCHEMA public
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO echovault_rw;

ALTER DEFAULT PRIVILEGES IN SCHEMA echovault
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO echovault_rw;

-- Create a function to verify pgvector is working
CREATE OR REPLACE FUNCTION pgvector_test()
RETURNS TABLE (
    test_result TEXT,
    vector_dimensions INTEGER,
    is_working BOOLEAN
)
LANGUAGE plpgsql
AS $$
DECLARE
    test_vector vector(3);
    dimensions INTEGER;
BEGIN
    -- Create a test vector
    test_vector := '[1,2,3]';
    
    -- Get the dimensions
    dimensions := array_length(test_vector::float8[], 1);
    
    -- Return result
    test_result := 'pgvector extension is working correctly';
    vector_dimensions := dimensions;
    is_working := TRUE;
    
    RETURN NEXT;
    
    EXCEPTION WHEN OTHERS THEN
        test_result := 'pgvector extension error: ' || SQLERRM;
        vector_dimensions := 0;
        is_working := FALSE;
        RETURN NEXT;
END;
$$;