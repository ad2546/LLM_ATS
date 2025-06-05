CREATE TABLE candidate (
    candidate_id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255),
    email VARCHAR(255),
    resume_path VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE category_score (
    score_id INT AUTO_INCREMENT PRIMARY KEY,
    candidate_id INT,
    category_name VARCHAR(255),
    score FLOAT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (candidate_id) REFERENCES candidate(candidate_id)
);
CREATE TABLE job_description (
    jd_id INT AUTO_INCREMENT PRIMARY KEY,
    jd_text TEXT,
    category_detected VARCHAR(255),
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE logs (
    log_id INT AUTO_INCREMENT PRIMARY KEY,
    log_type VARCHAR(50),            
    process VARCHAR(100),            
    message TEXT,                    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE `LLM_Resume`.`job_description` 
ADD COLUMN `key_responsibilities` VARCHAR(45) NULL AFTER `uploaded_at`,
ADD COLUMN `min_requirements` VARCHAR(45) NULL AFTER `key_responsibilities`;
