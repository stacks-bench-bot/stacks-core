// Copyright (C) 2013-2020 Blockstack PBC, a public benefit corporation
// Copyright (C) 2020-2026 Stacks Open Internet Foundation
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

use std::collections::HashMap;
use std::hash::Hash;

use stacks_common::types::StacksEpochId;
use stacks_common::types::chainstate::{StacksBlockId, TrieHash};
use stacks_common::util::hash::Sha512Trunc256Sum;

use super::clarity_store::SpecialCaseHandler;
use super::{ClarityBackingStore, ClarityDeserializable};
use crate::vm::Value;
use crate::vm::database::clarity_store::{ContractCommitment, make_contract_hash_key};
use crate::vm::errors::{VmExecutionError, VmInternalError};
use crate::vm::types::serialization::SerializationError;
use crate::vm::types::{QualifiedContractIdentifier, TypeSignature};

#[cfg(feature = "rollback_value_check")]
type RollbackValueCheck = String;
#[cfg(not(feature = "rollback_value_check"))]
type RollbackValueCheck = ();

#[cfg(not(feature = "rollback_value_check"))]
fn rollback_value_check(_value: &str, _check: &RollbackValueCheck) {}

#[cfg(not(feature = "rollback_value_check"))]
fn rollback_edits_push<T>(edits: &mut Vec<(T, RollbackValueCheck)>, key: T, _value: &str) {
    edits.push((key, ()));
}
// this function is used to check the lookup map when committing at the "bottom" of the
//   wrapper -- i.e., when committing to the underlying store. for the _unchecked_ implementation
//   this is used to get the edit _value_ out of the lookupmap, for used in the subsequent `put_all`
//   command.
#[cfg(not(feature = "rollback_value_check"))]
fn rollback_check_pre_bottom_commit<T>(
    edits: Vec<(T, RollbackValueCheck)>,
    lookup_map: &mut HashMap<T, Vec<String>>,
) -> Result<Vec<(T, String)>, VmInternalError>
where
    T: Eq + Hash + Clone,
{
    for (_, edit_history) in lookup_map.iter_mut() {
        edit_history.reverse();
    }

    let output = edits
        .into_iter()
        .map(|(key, _)| {
            let value = rollback_lookup_map(&key, &(), lookup_map)?;
            Ok((key, value))
        })
        .collect();

    assert!(lookup_map.is_empty());
    output
}

#[cfg(feature = "rollback_value_check")]
fn rollback_value_check(value: &String, check: &RollbackValueCheck) {
    assert_eq!(value, check)
}
#[cfg(feature = "rollback_value_check")]
fn rollback_edits_push<T>(edits: &mut Vec<(T, RollbackValueCheck)>, key: T, value: &str)
where
    T: Eq + Hash + Clone,
{
    edits.push((key, value.to_owned()));
}
// this function is used to check the lookup map when committing at the "bottom" of the
//   wrapper -- i.e., when committing to the underlying store.
#[cfg(feature = "rollback_value_check")]
fn rollback_check_pre_bottom_commit<T>(
    edits: Vec<(T, RollbackValueCheck)>,
    lookup_map: &mut HashMap<T, Vec<String>>,
) -> Result<Vec<(T, String)>, VmInternalError>
where
    T: Eq + Hash + Clone,
{
    for (_, edit_history) in lookup_map.iter_mut() {
        edit_history.reverse();
    }
    for (key, value) in edits.iter() {
        let _ = rollback_lookup_map(key, value, lookup_map);
    }
    assert!(lookup_map.is_empty());
    Ok(edits)
}

/// Result structure for fetched values from the
///  underlying store.
#[derive(Debug)]
pub struct ValueResult {
    pub value: Value,
    pub serialized_byte_len: u64,
}

pub struct RollbackContext {
    edits: Vec<(String, RollbackValueCheck)>,
    metadata_edits: Vec<((QualifiedContractIdentifier, String), RollbackValueCheck)>,
}

pub struct RollbackWrapper<'a> {
    // the underlying key-value storage.
    store: &'a mut dyn ClarityBackingStore,
    // lookup_map is a history of edits for a given key.
    //   in order of least-recent to most-recent at the tail.
    //   this allows ~ O(1) lookups, and ~ O(1) commits, roll-backs (amortized by # of PUTs).
    lookup_map: HashMap<String, Vec<String>>,
    metadata_lookup_map: HashMap<(QualifiedContractIdentifier, String), Vec<String>>,
    // stack keeps track of the most recent rollback context, which tells us which
    //   edits were performed by which context. at the moment, each context's edit history
    //   is a separate Vec which must be drained into the parent on commits, meaning that
    //   the amortized cost of committing a value isn't O(1), but actually O(k) where k is
    //   stack depth.
    //  TODO: The solution to this is to just have a _single_ edit stack, and merely store indexes
    //   to indicate a given contexts "start depth".
    stack: Vec<RollbackContext>,
    query_pending_data: bool,
    active_block_hash: Option<StacksBlockId>,
    materialized_read_cache: HashMap<StacksBlockId, HashMap<String, Option<String>>>,
    materialized_metadata_cache:
        HashMap<StacksBlockId, HashMap<(QualifiedContractIdentifier, String), Option<String>>>,
}

// This is used for preserving rollback data longer
//   than a BackingStore pointer. This is useful to prevent
//   a real mess of lifetime parameters in the database/context
//   and eval code.
pub struct RollbackWrapperPersistedLog {
    lookup_map: HashMap<String, Vec<String>>,
    metadata_lookup_map: HashMap<(QualifiedContractIdentifier, String), Vec<String>>,
    stack: Vec<RollbackContext>,
}

impl From<RollbackWrapper<'_>> for RollbackWrapperPersistedLog {
    fn from(o: RollbackWrapper<'_>) -> RollbackWrapperPersistedLog {
        RollbackWrapperPersistedLog {
            lookup_map: o.lookup_map,
            metadata_lookup_map: o.metadata_lookup_map,
            stack: o.stack,
        }
    }
}

impl Default for RollbackWrapperPersistedLog {
    fn default() -> Self {
        Self::new()
    }
}

impl RollbackWrapperPersistedLog {
    pub fn new() -> RollbackWrapperPersistedLog {
        RollbackWrapperPersistedLog {
            lookup_map: HashMap::new(),
            metadata_lookup_map: HashMap::new(),
            stack: Vec::new(),
        }
    }

    pub fn nest(&mut self) {
        self.stack.push(RollbackContext {
            edits: Vec::new(),
            metadata_edits: Vec::new(),
        });
    }
}

fn rollback_lookup_map<T>(
    key: &T,
    value: &RollbackValueCheck,
    lookup_map: &mut HashMap<T, Vec<String>>,
) -> Result<String, VmInternalError>
where
    T: Eq + Hash + Clone,
{
    let popped_value;
    let remove_edit_deque = {
        let key_edit_history = lookup_map.get_mut(key).ok_or_else(|| {
            VmInternalError::Expect(
                "ERROR: Clarity VM had edit log entry, but not lookup_map entry".into(),
            )
        })?;
        popped_value = key_edit_history.pop().ok_or_else(|| {
            VmInternalError::Expect("ERROR: expected value in edit history".into())
        })?;
        rollback_value_check(&popped_value, value);
        key_edit_history.is_empty()
    };
    if remove_edit_deque {
        lookup_map.remove(key);
    }
    Ok(popped_value)
}

impl<'a> RollbackWrapper<'a> {
    pub fn new(store: &'a mut dyn ClarityBackingStore) -> RollbackWrapper<'a> {
        RollbackWrapper {
            store,
            lookup_map: HashMap::new(),
            metadata_lookup_map: HashMap::new(),
            stack: Vec::new(),
            query_pending_data: true,
            active_block_hash: None,
            materialized_read_cache: HashMap::new(),
            materialized_metadata_cache: HashMap::new(),
        }
    }

    pub fn from_persisted_log(
        store: &'a mut dyn ClarityBackingStore,
        log: RollbackWrapperPersistedLog,
    ) -> RollbackWrapper<'a> {
        RollbackWrapper {
            store,
            lookup_map: log.lookup_map,
            metadata_lookup_map: log.metadata_lookup_map,
            stack: log.stack,
            query_pending_data: true,
            active_block_hash: None,
            materialized_read_cache: HashMap::new(),
            materialized_metadata_cache: HashMap::new(),
        }
    }

    pub fn get_cc_special_cases_handler(&self) -> Option<SpecialCaseHandler> {
        self.store.get_cc_special_cases_handler()
    }

    pub fn nest(&mut self) {
        self.stack.push(RollbackContext {
            edits: Vec::new(),
            metadata_edits: Vec::new(),
        });
    }

    // Rollback the child's edits.
    //   this clears all edits from the child's edit queue,
    //     and removes any of those edits from the lookup map.
    pub fn rollback(&mut self) -> Result<(), VmInternalError> {
        let mut last_item = self.stack.pop().ok_or_else(|| {
            VmInternalError::Expect("ERROR: Clarity VM attempted to commit past the stack.".into())
        })?;

        last_item.edits.reverse();
        last_item.metadata_edits.reverse();

        for (key, value) in last_item.edits.drain(..) {
            rollback_lookup_map(&key, &value, &mut self.lookup_map)?;
        }

        for (key, value) in last_item.metadata_edits.drain(..) {
            rollback_lookup_map(&key, &value, &mut self.metadata_lookup_map)?;
        }

        Ok(())
    }

    pub fn depth(&self) -> usize {
        self.stack.len()
    }

    pub fn commit(&mut self) -> Result<(), VmInternalError> {
        let mut last_item = self.stack.pop().ok_or_else(|| {
            VmInternalError::Expect("ERROR: Clarity VM attempted to commit past the stack.".into())
        })?;

        if let Some(next_up) = self.stack.last_mut() {
            // bubble up to the next item in the stack
            // last_mut() must exist because of the if-statement
            for (key, value) in last_item.edits.drain(..) {
                next_up.edits.push((key, value));
            }
            for (key, value) in last_item.metadata_edits.drain(..) {
                next_up.metadata_edits.push((key, value));
            }
        } else {
            // stack is empty, committing to the backing store
            let all_edits =
                rollback_check_pre_bottom_commit(last_item.edits, &mut self.lookup_map)?;
            if !all_edits.is_empty() {
                self.store.put_all_data(all_edits).map_err(|e| {
                    VmInternalError::Expect(format!(
                        "ERROR: Failed to commit data to sql store: {e:?}"
                    ))
                })?;
                self.materialized_read_cache.clear();
            }

            let metadata_edits = rollback_check_pre_bottom_commit(
                last_item.metadata_edits,
                &mut self.metadata_lookup_map,
            )?;
            if !metadata_edits.is_empty() {
                self.store.put_all_metadata(metadata_edits).map_err(|e| {
                    VmInternalError::Expect(format!(
                        "ERROR: Failed to commit data to sql store: {e:?}"
                    ))
                })?;
                self.materialized_metadata_cache.clear();
            }
        }

        Ok(())
    }
}

fn inner_put_data<T>(
    lookup_map: &mut HashMap<T, Vec<String>>,
    edits: &mut Vec<(T, RollbackValueCheck)>,
    key: T,
    value: String,
) where
    T: Eq + Hash + Clone,
{
    let key_edit_deque = lookup_map.entry(key.clone()).or_default();
    rollback_edits_push(edits, key, &value);
    key_edit_deque.push(value);
}

impl RollbackWrapper<'_> {
    pub fn put_data(&mut self, key: &str, value: &str) -> Result<(), VmExecutionError> {
        let current = self.stack.last_mut().ok_or_else(|| {
            VmInternalError::Expect("ERROR: Clarity VM attempted PUT on non-nested context.".into())
        })?;

        inner_put_data(
            &mut self.lookup_map,
            &mut current.edits,
            key.to_string(),
            value.to_string(),
        );
        Ok(())
    }

    ///
    /// `query_pending_data` indicates whether the rollback wrapper should query the rollback
    ///    wrapper's pending data on reads. This is set to `false` during (at-block ...) closures,
    ///    and `true` otherwise.
    ///
    pub fn set_block_hash(
        &mut self,
        bhh: StacksBlockId,
        query_pending_data: bool,
    ) -> Result<StacksBlockId, VmExecutionError> {
        self.store.set_block_hash(bhh.clone()).inspect(|_| {
            // use and_then so that query_pending_data is only set once set_block_hash succeeds
            //  this doesn't matter in practice, because a set_block_hash failure always aborts
            //  the transaction with a runtime error (destroying its environment), but it's much
            //  better practice to do this, especially if the abort behavior changes in the future.
            self.query_pending_data = query_pending_data;
            self.active_block_hash = Some(bhh);
        })
    }

    fn get_materialized_data(&mut self, key: &str) -> Result<Option<String>, VmExecutionError> {
        if self.query_pending_data {
            return self.store.get_data(key);
        }

        let Some(active_block_hash) = self.active_block_hash.as_ref() else {
            return self.store.get_data(key);
        };

        if let Some(value) = self
            .materialized_read_cache
            .get(active_block_hash)
            .and_then(|block_cache| block_cache.get(key))
        {
            return Ok(value.clone());
        }

        let value = self.store.get_data(key)?;
        self.materialized_read_cache
            .entry(active_block_hash.clone())
            .or_default()
            .insert(key.to_string(), value.clone());
        Ok(value)
    }

    fn get_materialized_metadata(
        &mut self,
        contract: &QualifiedContractIdentifier,
        key: &str,
    ) -> Result<Option<String>, VmExecutionError> {
        if self.query_pending_data {
            return self.store.get_metadata(contract, key);
        }

        let Some(active_block_hash) = self.active_block_hash.as_ref() else {
            return self.store.get_metadata(contract, key);
        };

        let metadata_key = (contract.clone(), key.to_string());
        if let Some(value) = self
            .materialized_metadata_cache
            .get(active_block_hash)
            .and_then(|block_cache| block_cache.get(&metadata_key))
        {
            return Ok(value.clone());
        }

        let value = self.store.get_metadata(contract, key)?;
        self.materialized_metadata_cache
            .entry(active_block_hash.clone())
            .or_default()
            .insert(metadata_key, value.clone());
        Ok(value)
    }

    /// this function will only return commitment proofs for values _already_ materialized
    ///  in the underlying store. otherwise it returns None.
    #[cfg_attr(feature = "profiler", stacks_profiler::profile)]
    pub fn get_data_with_proof<T>(
        &mut self,
        key: &str,
    ) -> Result<Option<(T, Vec<u8>)>, VmExecutionError>
    where
        T: ClarityDeserializable<T>,
    {
        self.store
            .get_data_with_proof(key)?
            .map(|(value, proof)| Ok((T::deserialize(&value)?, proof)))
            .transpose()
    }

    /// this function will only return commitment proofs for values _already_ materialized
    ///  in the underlying store. otherwise it returns None.
    #[cfg_attr(feature = "profiler", stacks_profiler::profile)]
    pub fn get_data_with_proof_by_hash<T>(
        &mut self,
        hash: &TrieHash,
    ) -> Result<Option<(T, Vec<u8>)>, VmExecutionError>
    where
        T: ClarityDeserializable<T>,
    {
        self.store
            .get_data_with_proof_from_path(hash)?
            .map(|(value, proof)| Ok((T::deserialize(&value)?, proof)))
            .transpose()
    }

    #[cfg_attr(feature = "profiler", stacks_profiler::profile)]
    pub fn get_data<T>(&mut self, key: &str) -> Result<Option<T>, VmExecutionError>
    where
        T: ClarityDeserializable<T>,
    {
        self.stack.last().ok_or_else(|| {
            VmInternalError::Expect("ERROR: Clarity VM attempted GET on non-nested context.".into())
        })?;

        if self.query_pending_data
            && let Some(pending_value) = self.lookup_map.get(key).and_then(|x| x.last())
        {
            // if there's pending data and we're querying pending data, return here
            return Some(T::deserialize(pending_value)).transpose();
        }
        // otherwise, lookup from store
        self.get_materialized_data(key)?
            .map(|x| T::deserialize(&x))
            .transpose()
    }

    /// DO NOT USE IN CONSENSUS CODE.
    ///
    /// Load data directly from the underlying store, given its trie hash.  The lookup map will not
    /// be used.
    ///
    /// This should never be called from within the Clarity VM, or via block-processing.  It's only
    /// meant to be used by the RPC system.
    pub fn get_data_by_hash<T>(&mut self, hash: &TrieHash) -> Result<Option<T>, VmExecutionError>
    where
        T: ClarityDeserializable<T>,
    {
        self.store
            .get_data_from_path(hash)?
            .map(|x| T::deserialize(&x))
            .transpose()
    }

    pub fn deserialize_value(
        value_hex: &str,
        expected: &TypeSignature,
        epoch: &StacksEpochId,
    ) -> Result<ValueResult, SerializationError> {
        let serialized_byte_len = value_hex.len() as u64 / 2;
        let sanitize = epoch.value_sanitizing();
        let value = Value::try_deserialize_hex(value_hex, expected, sanitize)?;

        Ok(ValueResult {
            value,
            serialized_byte_len,
        })
    }

    /// Get a Clarity value from the underlying Clarity KV store.
    /// Returns Some if found, with the Clarity Value and the serialized byte length of the value.
    #[cfg_attr(feature = "profiler", stacks_profiler::profile)]
    pub fn get_value(
        &mut self,
        key: &str,
        expected: &TypeSignature,
        epoch: &StacksEpochId,
    ) -> Result<Option<ValueResult>, SerializationError> {
        self.stack.last().ok_or_else(|| {
            SerializationError::DeserializationFailure(
                "ERROR: Clarity VM attempted GET on non-nested context.".into(),
            )
        })?;

        if self.query_pending_data
            && let Some(x) = self.lookup_map.get(key).and_then(|x| x.last())
        {
            return Ok(Some(Self::deserialize_value(x, expected, epoch)?));
        }
        let stored_data = self.get_materialized_data(key).map_err(|_| {
            SerializationError::DeserializationFailure(
                "ERROR: Clarity backing store failure".into(),
            )
        })?;
        match stored_data {
            Some(x) => Ok(Some(Self::deserialize_value(&x, expected, epoch)?)),
            None => Ok(None),
        }
    }

    /// This is the height we are currently constructing. It comes from the MARF.
    pub fn get_current_block_height(&mut self) -> u32 {
        self.store.get_current_block_height()
    }

    /// Is None if `block_height` >= the "currently" under construction Stacks block height.
    pub fn get_block_header_hash(&mut self, block_height: u32) -> Option<StacksBlockId> {
        self.store.get_block_at_height(block_height)
    }

    #[cfg_attr(feature = "profiler", stacks_profiler::profile)]
    pub fn get_contract_hash(
        &mut self,
        contract: &QualifiedContractIdentifier,
    ) -> Result<Option<Sha512Trunc256Sum>, VmExecutionError> {
        let key = make_contract_hash_key(contract);
        let s = match self.get_data::<String>(&key)? {
            Some(s) => s,
            None => return Ok(None),
        };
        let cc = ContractCommitment::deserialize(&s)?;
        Ok(Some(cc.hash))
    }

    #[cfg_attr(feature = "profiler", stacks_profiler::profile)]
    pub fn prepare_for_contract_metadata(
        &mut self,
        contract: &QualifiedContractIdentifier,
        content_hash: Sha512Trunc256Sum,
    ) -> Result<(), VmExecutionError> {
        let key = make_contract_hash_key(contract);
        let value = self.store.make_contract_commitment(content_hash);
        self.put_data(&key, &value)
    }

    #[cfg_attr(feature = "profiler", stacks_profiler::profile)]
    pub fn insert_metadata(
        &mut self,
        contract: &QualifiedContractIdentifier,
        key: &str,
        value: &str,
    ) -> Result<(), VmInternalError> {
        let current = self.stack.last_mut().ok_or_else(|| {
            VmInternalError::Expect("ERROR: Clarity VM attempted PUT on non-nested context.".into())
        })?;

        let metadata_key = (contract.clone(), key.to_string());

        inner_put_data(
            &mut self.metadata_lookup_map,
            &mut current.metadata_edits,
            metadata_key,
            value.to_string(),
        );
        Ok(())
    }

    // Throws a NoSuchContract error if contract doesn't exist,
    //   returns None if there is no such metadata field.
    #[cfg_attr(feature = "profiler", stacks_profiler::profile)]
    pub fn get_metadata(
        &mut self,
        contract: &QualifiedContractIdentifier,
        key: &str,
    ) -> Result<Option<String>, VmExecutionError> {
        self.stack.last().ok_or_else(|| {
            VmInternalError::Expect("ERROR: Clarity VM attempted GET on non-nested context.".into())
        })?;

        // This is THEORETICALLY a spurious clone, but it's hard to turn something like
        //  (&A, &B) into &(A, B).
        let metadata_key = (contract.clone(), key.to_string());
        let lookup_result = if self.query_pending_data {
            self.metadata_lookup_map
                .get(&metadata_key)
                .and_then(|x| x.last().cloned())
        } else {
            None
        };

        match lookup_result {
            Some(x) => Ok(Some(x)),
            None => self.get_materialized_metadata(contract, key),
        }
    }

    // Throws a NoSuchContract error if contract doesn't exist,
    //   returns None if there is no such metadata field.
    #[cfg_attr(feature = "profiler", stacks_profiler::profile)]
    pub fn get_metadata_manual(
        &mut self,
        at_height: u32,
        contract: &QualifiedContractIdentifier,
        key: &str,
    ) -> Result<Option<String>, VmExecutionError> {
        self.stack.last().ok_or_else(|| {
            VmInternalError::Expect("ERROR: Clarity VM attempted GET on non-nested context.".into())
        })?;

        // This is THEORETICALLY a spurious clone, but it's hard to turn something like
        //  (&A, &B) into &(A, B).
        let metadata_key = (contract.clone(), key.to_string());
        let lookup_result = if self.query_pending_data {
            self.metadata_lookup_map
                .get(&metadata_key)
                .and_then(|x| x.last().cloned())
        } else {
            None
        };

        match lookup_result {
            Some(x) => Ok(Some(x)),
            None => self.store.get_metadata_manual(at_height, contract, key),
        }
    }

    #[cfg_attr(feature = "profiler", stacks_profiler::profile)]
    pub fn has_entry(&mut self, key: &str) -> Result<bool, VmExecutionError> {
        self.stack.last().ok_or_else(|| {
            VmInternalError::Expect("ERROR: Clarity VM attempted GET on non-nested context.".into())
        })?;
        if self.query_pending_data && self.lookup_map.contains_key(key) {
            Ok(true)
        } else {
            self.store.has_entry(key)
        }
    }

    #[cfg_attr(feature = "profiler", stacks_profiler::profile)]
    pub fn has_metadata_entry(
        &mut self,
        contract: &QualifiedContractIdentifier,
        key: &str,
    ) -> bool {
        matches!(self.get_metadata(contract, key), Ok(Some(_)))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use stacks_common::types::StacksEpochId;
    use stacks_common::types::chainstate::{StacksBlockId, TrieHash};
    use stacks_common::util::hash::{Sha512Trunc256Sum, to_hex};

    use super::RollbackWrapper;
    use crate::vm::Value;
    use crate::vm::database::ClarityBackingStore;
    use crate::vm::errors::VmExecutionError;
    use crate::vm::types::{QualifiedContractIdentifier, TypeSignature};

    struct CountingBackingStore {
        active_block_hash: StacksBlockId,
        data: HashMap<StacksBlockId, HashMap<String, String>>,
        metadata: HashMap<StacksBlockId, HashMap<(QualifiedContractIdentifier, String), String>>,
        data_reads: HashMap<(StacksBlockId, String), usize>,
        metadata_reads: HashMap<(StacksBlockId, QualifiedContractIdentifier, String), usize>,
    }

    impl CountingBackingStore {
        fn new(active_block_hash: StacksBlockId) -> Self {
            Self {
                active_block_hash,
                data: HashMap::new(),
                metadata: HashMap::new(),
                data_reads: HashMap::new(),
                metadata_reads: HashMap::new(),
            }
        }

        fn insert_data(&mut self, block_hash: StacksBlockId, key: &str, value: &str) {
            self.data
                .entry(block_hash)
                .or_default()
                .insert(key.to_string(), value.to_string());
        }

        fn insert_metadata(
            &mut self,
            block_hash: StacksBlockId,
            contract: QualifiedContractIdentifier,
            key: &str,
            value: &str,
        ) {
            self.metadata
                .entry(block_hash)
                .or_default()
                .insert((contract, key.to_string()), value.to_string());
        }

        fn data_read_count(&self, block_hash: &StacksBlockId, key: &str) -> usize {
            self.data_reads
                .get(&(block_hash.clone(), key.to_string()))
                .copied()
                .unwrap_or(0)
        }

        fn metadata_read_count(
            &self,
            block_hash: &StacksBlockId,
            contract: &QualifiedContractIdentifier,
            key: &str,
        ) -> usize {
            self.metadata_reads
                .get(&(block_hash.clone(), contract.clone(), key.to_string()))
                .copied()
                .unwrap_or(0)
        }
    }

    impl ClarityBackingStore for CountingBackingStore {
        fn put_all_data(&mut self, items: Vec<(String, String)>) -> Result<(), VmExecutionError> {
            let data = self.data.entry(self.active_block_hash.clone()).or_default();
            for (key, value) in items {
                data.insert(key, value);
            }
            Ok(())
        }

        fn get_data(&mut self, key: &str) -> Result<Option<String>, VmExecutionError> {
            *self
                .data_reads
                .entry((self.active_block_hash.clone(), key.to_string()))
                .or_default() += 1;
            Ok(self
                .data
                .get(&self.active_block_hash)
                .and_then(|data| data.get(key).cloned()))
        }

        fn get_data_from_path(
            &mut self,
            _hash: &TrieHash,
        ) -> Result<Option<String>, VmExecutionError> {
            Ok(None)
        }

        fn get_data_with_proof(
            &mut self,
            key: &str,
        ) -> Result<Option<(String, Vec<u8>)>, VmExecutionError> {
            Ok(self.get_data(key)?.map(|value| (value, Vec::new())))
        }

        fn get_data_with_proof_from_path(
            &mut self,
            _hash: &TrieHash,
        ) -> Result<Option<(String, Vec<u8>)>, VmExecutionError> {
            Ok(None)
        }

        fn set_block_hash(
            &mut self,
            bhh: StacksBlockId,
        ) -> Result<StacksBlockId, VmExecutionError> {
            let prior = self.active_block_hash.clone();
            self.active_block_hash = bhh;
            Ok(prior)
        }

        fn get_block_at_height(&mut self, _height: u32) -> Option<StacksBlockId> {
            None
        }

        fn get_current_block_height(&mut self) -> u32 {
            0
        }

        fn get_open_chain_tip_height(&mut self) -> u32 {
            0
        }

        fn get_open_chain_tip(&mut self) -> StacksBlockId {
            self.active_block_hash.clone()
        }

        fn get_contract_hash(
            &mut self,
            _contract: &QualifiedContractIdentifier,
        ) -> Result<(StacksBlockId, Sha512Trunc256Sum), VmExecutionError> {
            Ok((self.active_block_hash.clone(), Sha512Trunc256Sum([0; 32])))
        }

        #[cfg(feature = "rusqlite")]
        fn get_side_store(&mut self) -> &rusqlite::Connection {
            panic!("CountingBackingStore does not use sqlite")
        }

        fn insert_metadata(
            &mut self,
            contract: &QualifiedContractIdentifier,
            key: &str,
            value: &str,
        ) -> Result<(), VmExecutionError> {
            self.metadata
                .entry(self.active_block_hash.clone())
                .or_default()
                .insert((contract.clone(), key.to_string()), value.to_string());
            Ok(())
        }

        fn get_metadata(
            &mut self,
            contract: &QualifiedContractIdentifier,
            key: &str,
        ) -> Result<Option<String>, VmExecutionError> {
            *self
                .metadata_reads
                .entry((
                    self.active_block_hash.clone(),
                    contract.clone(),
                    key.to_string(),
                ))
                .or_default() += 1;
            Ok(self
                .metadata
                .get(&self.active_block_hash)
                .and_then(|data| data.get(&(contract.clone(), key.to_string())).cloned()))
        }

        fn get_metadata_manual(
            &mut self,
            _at_height: u32,
            contract: &QualifiedContractIdentifier,
            key: &str,
        ) -> Result<Option<String>, VmExecutionError> {
            self.get_metadata(contract, key)
        }
    }

    #[test]
    fn materialized_reads_are_cached_by_block_hash() {
        let block_a = StacksBlockId([1; 32]);
        let block_b = StacksBlockId([2; 32]);
        let mut store = CountingBackingStore::new(block_a.clone());
        store.insert_data(block_a.clone(), "key", "block-a");
        store.insert_data(block_b.clone(), "key", "block-b");

        {
            let mut wrapper = RollbackWrapper::new(&mut store);
            wrapper.nest();

            wrapper.set_block_hash(block_a.clone(), false).unwrap();
            assert_eq!(
                wrapper.get_data::<String>("key").unwrap(),
                Some("block-a".to_string())
            );
            assert_eq!(
                wrapper.get_data::<String>("key").unwrap(),
                Some("block-a".to_string())
            );

            wrapper.set_block_hash(block_b.clone(), false).unwrap();
            assert_eq!(
                wrapper.get_data::<String>("key").unwrap(),
                Some("block-b".to_string())
            );

            wrapper.set_block_hash(block_a.clone(), false).unwrap();
            assert_eq!(
                wrapper.get_data::<String>("key").unwrap(),
                Some("block-a".to_string())
            );
        }

        assert_eq!(store.data_read_count(&block_a, "key"), 1);
        assert_eq!(store.data_read_count(&block_b, "key"), 1);
    }

    #[test]
    fn materialized_read_cache_ignores_surrounding_pending_writes() {
        let block = StacksBlockId([3; 32]);
        let mut store = CountingBackingStore::new(block.clone());
        store.insert_data(block.clone(), "key", "committed");

        {
            let mut wrapper = RollbackWrapper::new(&mut store);
            wrapper.nest();

            wrapper.set_block_hash(block.clone(), false).unwrap();
            assert_eq!(
                wrapper.get_data::<String>("key").unwrap(),
                Some("committed".to_string())
            );

            wrapper.set_block_hash(block.clone(), true).unwrap();
            wrapper.put_data("key", "pending").unwrap();
            assert_eq!(
                wrapper.get_data::<String>("key").unwrap(),
                Some("pending".to_string())
            );

            wrapper.set_block_hash(block.clone(), false).unwrap();
            assert_eq!(
                wrapper.get_data::<String>("key").unwrap(),
                Some("committed".to_string())
            );

            wrapper.set_block_hash(block.clone(), true).unwrap();
            assert_eq!(
                wrapper.get_data::<String>("key").unwrap(),
                Some("pending".to_string())
            );
        }

        assert_eq!(store.data_read_count(&block, "key"), 1);
    }

    #[test]
    fn materialized_cache_is_used_by_value_reads() {
        let block = StacksBlockId([4; 32]);
        let mut store = CountingBackingStore::new(block.clone());
        let serialized = to_hex(
            &Value::Int(123)
                .serialize_to_vec()
                .expect("failed to serialize test value"),
        );
        store.insert_data(block.clone(), "value-key", &serialized);

        {
            let mut wrapper = RollbackWrapper::new(&mut store);
            wrapper.nest();
            wrapper.set_block_hash(block.clone(), false).unwrap();

            let first = wrapper
                .get_value(
                    "value-key",
                    &TypeSignature::IntType,
                    &StacksEpochId::Epoch25,
                )
                .unwrap()
                .unwrap();
            let second = wrapper
                .get_value(
                    "value-key",
                    &TypeSignature::IntType,
                    &StacksEpochId::Epoch25,
                )
                .unwrap()
                .unwrap();

            assert_eq!(first.value, Value::Int(123));
            assert_eq!(second.value, Value::Int(123));
            assert_eq!(first.serialized_byte_len, second.serialized_byte_len);
        }

        assert_eq!(store.data_read_count(&block, "value-key"), 1);
    }

    #[test]
    fn materialized_metadata_reads_are_cached_by_block_hash() {
        let block_a = StacksBlockId([5; 32]);
        let block_b = StacksBlockId([6; 32]);
        let contract = QualifiedContractIdentifier::transient();
        let mut store = CountingBackingStore::new(block_a.clone());
        store.insert_metadata(block_a.clone(), contract.clone(), "meta-key", "meta-a");
        store.insert_metadata(block_b.clone(), contract.clone(), "meta-key", "meta-b");

        {
            let mut wrapper = RollbackWrapper::new(&mut store);
            wrapper.nest();

            wrapper.set_block_hash(block_a.clone(), false).unwrap();
            assert_eq!(
                wrapper.get_metadata(&contract, "meta-key").unwrap(),
                Some("meta-a".to_string())
            );
            assert_eq!(
                wrapper.get_metadata(&contract, "meta-key").unwrap(),
                Some("meta-a".to_string())
            );

            wrapper.set_block_hash(block_b.clone(), false).unwrap();
            assert_eq!(
                wrapper.get_metadata(&contract, "meta-key").unwrap(),
                Some("meta-b".to_string())
            );
        }

        assert_eq!(
            store.metadata_read_count(&block_a, &contract, "meta-key"),
            1
        );
        assert_eq!(
            store.metadata_read_count(&block_b, &contract, "meta-key"),
            1
        );
    }
}
