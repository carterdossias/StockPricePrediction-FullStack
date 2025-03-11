CREATE TABLE users (
    user_id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    last_login TIMESTAMP NULL,
    account_balance DECIMAL(15,2) DEFAULT 0.00,
    available_funds DECIMAL(15,2) DEFAULT 0.00,
    portfolio_value DECIMAL(15,2) DEFAULT 0.00,
    risk_profile VARCHAR(20) DEFAULT 'moderate',
    account_status INT DEFAULT 1
);