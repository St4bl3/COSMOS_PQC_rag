# COSMOS: Post-Quantum Secure Communication Module

## Overview

This repository contains the **Post-Quantum Cryptography (PQC) Security Module**, a specialized component designed for the larger **COSMOS** multi-agent project. The sole purpose of this module is to provide a robust, forward-secure communication layer that protects all inter-agent messages from both classical and future quantum cryptographic threats.

This module is a critical dependency for other repositories in the COSMOS ecosystem, ensuring the **confidentiality**, **integrity**, and **authenticity** of data exchanged between all agents (e.g., the Orchestrator, SOPAgent, CodeGenAgent). This implementation is submitted as a part of the Computer Security coursework.

---

## The Quantum Threat ⚛️

Classical public-key cryptographic algorithms, such as RSA and ECC, are vulnerable to attacks from future, large-scale quantum computers. As the COSMOS project is designed for long-term, sensitive operations, it is imperative to secure its internal communications against this emerging threat. This module directly addresses that challenge by implementing a **hybrid cryptographic scheme** based on algorithms selected by the U.S. National Institute of Standards and Technology (NIST) for standardization.

---

## Core Security Features 🔐

* **Post-Quantum Cryptography (PQC)**: Utilizes a state-of-the-art PQC-hybrid scheme to ensure resilience against quantum attacks.
    * **Key Encapsulation Mechanism (KEM)**: **Kyber768** is used for securely establishing shared secrets.
    * **Digital Signature Algorithm**: **Dilithium3** is used to guarantee the authenticity and integrity of messages.

* **Hybrid Encryption (KEM + DEM)**: To combine the security of asymmetric PQC with the efficiency of symmetric encryption, the module uses a standard KEM + DEM model.
    1.  A secure, one-time **AES-256-GCM** symmetric key is generated to encrypt the actual message payload (the DEM).
    2.  This AES key is then securely encapsulated using the recipient's public **Kyber768** key (the KEM).

* **`SecureEnvelope` Wrapper**: All messages are encapsulated within a `SecureEnvelope` object. This standardized data structure contains the encrypted payload, the encapsulated key, the sender's digital signature, and other critical metadata, ensuring that all necessary security components are transmitted as a single, atomic unit.

* **Centralized Key Management**: A `KeyManager` class handles the generation, storage, and retrieval of PQC key pairs (Kyber and Dilithium) for all registered agents, simplifying the key lifecycle within the broader COSMOS project.

---

## Architectural Workflow & Integration

This security module is designed to be seamlessly integrated as a library into the main COSMOS application workflow.

1.  **Initialization**: In the main COSMOS application, the `KeyManager` is invoked first to generate and load PQC key pairs for all agents.

2.  **Message Encryption & Signing**: When one agent (the "sender") needs to send a message to another (the "recipient"):
    a. The sender invokes the `SecureCommunicator`'s `encrypt_and_sign` method, passing the plaintext message and the recipient's ID.
    b. The `SecureCommunicator` encrypts the message payload using AES-256-GCM and encapsulates the AES key with the recipient's public Kyber key.
    c. It then signs a digest of the message components with its own private Dilithium key.
    d. All these components are packed into a `SecureEnvelope` object.

3.  **Transmission**: The serialized `SecureEnvelope` is transmitted over the network to the recipient agent.

4.  **Message Verification & Decryption**: Upon receiving a `SecureEnvelope`, the recipient agent:
    a. Uses the sender's public Dilithium key to **verify the digital signature**. This confirms the message's authenticity and ensures it has not been tampered with in transit.
    b. If verification succeeds, it uses its own private Kyber key to **decapsulate the AES session key**.
    c. Finally, it uses the AES key to **decrypt the message payload**, retrieving the original plaintext.

---

## How to Use This Module

This module is intended to be installed as a package in other COSMOS project repositories.

### Installation

```bash
# (Assuming the package is hosted in a git repository)
pip install git+[https://github.com/your-username/cosmos_security_module.git](https://github.com/your-username/cosmos_security_module.git)

# Or install from a local path
pip install -e .
